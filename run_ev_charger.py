"""
EV Solar Smart Charging - Runnable script

This script loads (or generates) a small synthetic dataset of solar + EV charging
and trains a lightweight LSTM to predict normalized solar irradiance. It then
simulates a simple load-balancing decision: if predicted irradiance > threshold
then charge from solar, else from grid.

Run:
  python run_ev_charger.py

Optional: place `ev_solar_project_dataset.csv` in the same folder to use real data.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
import argparse

# Check for TensorFlow availability; if not present, we'll fall back to sklearn
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def generate_synthetic(n_days=120, seed=42):
    np.random.seed(seed)
    dates = pd.date_range(start="2025-01-01", periods=n_days, freq="D")
    data = {
        "date": dates,
        "solar_irradiance": np.random.randint(150, 900, n_days),
        "cloud_cover_percent": np.random.randint(0, 100, n_days),
        "rainfall_mm": np.random.randint(0, 50, n_days),
        "ev_charging_sessions": np.random.randint(5, 30, n_days),
        "energy_consumed_kWh": np.random.randint(40, 300, n_days),
    }
    return pd.DataFrame(data)


def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length, 0])  # predict solar_irradiance (scaled)
    return np.array(X), np.array(y)


if TF_AVAILABLE:
    def build_lstm(seq_length, n_features):
        model = Sequential([
            LSTM(32, input_shape=(seq_length, n_features)),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mae")
        return model


def main(args):
    # Load or generate dataset
    csv_path = os.path.join(os.path.dirname(__file__), "ev_solar_project_dataset.csv")
    if os.path.exists(csv_path):
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["date"])
    else:
        print("No CSV found — generating synthetic dataset.")
        df = generate_synthetic(n_days=140)

    print(df.head())

    # Preprocessing
    df = df.sort_values("date").reset_index(drop=True)
    df = df.fillna(method="ffill")

    features = ["solar_irradiance", "cloud_cover_percent", "rainfall_mm"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    seq_length = 7
    X, y = create_sequences(scaled, seq_length=seq_length)

    # Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    # Choose training approach depending on TensorFlow availability
    if TF_AVAILABLE:
        model = build_lstm(seq_length, X.shape[2])
        # Train briefly for a quick smoke test (small epochs). Increase epochs for real training.
        epochs = max(3, args.epochs)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=8, validation_split=0.15, verbose=1)

        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (normalized, TF LSTM): {mae:.4f}")

        # Save TF model
        model_path = os.path.join(os.path.dirname(__file__), "solar_predictor.h5")
        model.save(model_path)
        print(f"Saved TensorFlow model to {model_path}")
    else:
        # Flatten sequences to fixed-size features for scikit-learn
        def flatten_sequences(Xarr):
            flat = []
            for seq in Xarr:
                stats = np.concatenate([np.mean(seq, axis=0), np.std(seq, axis=0), seq[-1]])
                flat.append(stats)
            return np.array(flat)

        X_train_flat = flatten_sequences(X_train)
        X_test_flat = flatten_sequences(X_test)

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train)
        y_pred = rf.predict(X_test_flat)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (normalized, RandomForest): {mae:.4f}")

        # Save sklearn model
        model_path = os.path.join(os.path.dirname(__file__), "solar_predictor_rf.joblib")
        joblib.dump(rf, model_path)
        print(f"Saved RandomForest model to {model_path}")

    # Optional plotting (works for both approaches)
    if args.plot:
        plt.figure(figsize=(10, 4))
        plt.plot(y_test, label="Actual (scaled)")
        plt.plot(y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred, label="Predicted (scaled)")
        plt.legend()
        plt.title("Solar Irradiance Prediction (Scaled)")
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "prediction_plot.png"))
        print("Saved plot to prediction_plot.png")

    # Load balancing simulation
    solar_threshold = args.threshold  # on scaled irradiance [0..1]
    solar_usage = (np.array(y_pred).flatten() > solar_threshold).sum()
    grid_usage = len(y_pred) - solar_usage

    print("\nSimulation Results:")
    print(f"Solar-powered sessions (predicted > {solar_threshold}): {solar_usage}")
    print(f"Grid-powered sessions: {grid_usage}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EV Solar smart-charging demo")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (small by default)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Scaled irradiance threshold [0..1] to prefer solar")
    parser.add_argument("--plot", action="store_true", help="Save prediction plot as PNG")
    args = parser.parse_args()

    # Limit TensorFlow logging for clarity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main(args)
