# 💳 Credit Card Fraud Detection

A machine learning pipeline to detect fraudulent credit card transactions 
using real anonymized data from a European bank.

The key challenge: only **0.17% of transactions are fraudulent** — 
standard models fail completely on such imbalanced data.

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| [01 — Fraud Detection](01_fraud_detection.ipynb) | EDA, SMOTE, XGBoost, threshold tuning |
| [02 — Model Explainability](02_model_explainability.ipynb) | SHAP, Cross-Validation, Precision-Recall |
| [03 — FHE Privacy Preserving](03_fhe_privacy_preserving.ipynb) | TenSEAL CKKS |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| F1 Score | **0.851** |
| ROC-AUC | **0.979** |
| Precision | 0.856 |
| Recall | 0.847 |

Out of 98 fraud cases in the test set:
- **83 caught** — customers protected
- **14 false alarms** — legitimate customers wrongly flagged
- **15 missed** — unavoidable at current threshold

---

## 🔍 Approach

### The Imbalance Problem
A naive model predicting "all legitimate" achieves 99.83% accuracy 
but detects zero fraud. We solve this with:
- **SMOTE** — synthetic oversampling of fraud cases to 50/50 balance
- **Proper metrics** — Precision, Recall, F1, ROC-AUC (not accuracy)

### Models Compared
| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.058 | 0.918 | 0.109 | 0.970 |
| LightGBM | 0.500 | 0.888 | 0.640 | 0.969 |
| **XGBoost** | **0.731** | **0.888** | **0.802** | **0.979** |
| **XGBoost + Threshold Tuning** | **0.856** | **0.847** | **0.851** | **0.979** |

### Threshold Tuning
Default classification threshold (0.50) was raised to **0.86** — 
the model only flags a transaction as fraud when 86%+ confident.
This improved F1 from 0.802 to **0.851**.

---

## 📁 Repository Structure
```
fraud-detection-analysis/
│
├── data/
│   └── creditcard.csv              # raw dataset (download from Kaggle)
│
├── visuals/
│   ├── eda_overview.png            # class imbalance + amount + time
│   ├── threshold_tuning.png        # precision/recall vs threshold
│   ├── final_evaluation.png        # confusion matrix + ROC curve
│   ├── shap_global.png             # global feature importance (SHAP)
│   ├── shap_local.png              # local transaction explanations
│   └── precision_recall.png        # PR curve + optimal threshold
│
├── models/
│   ├── xgb_fraud_model.pkl         # trained XGBoost model
│   └── optimal_threshold.pkl       # best classification threshold
│
├── 01_fraud_detection.ipynb        # EDA, SMOTE, modeling, tuning
├── 02_model_explainability.ipynb   # SHAP, CV, Precision-Recall
└── README.md
```

---

## 💡 Key Insights

1. **Class imbalance is the core challenge** — SMOTE and proper metrics 
   are essential for any fraud detection system
2. **XGBoost outperforms** simpler models significantly in imbalanced settings
3. **Threshold tuning matters** — raising threshold from 0.50 to 0.86 
   improved F1 by 6% with minimal Recall loss
4. **The 15 missed fraud cases** represent a business decision, not just 
   a technical limitation — optimal threshold depends on the cost of 
   fraud vs cost of blocking legitimate customers

---

## 🛠️ Tech Stack

- **Python 3.12**
- **pandas / numpy** — data manipulation
- **matplotlib / seaborn** — visualization
- **scikit-learn** — model evaluation, SMOTE pipeline
- **imbalanced-learn** — SMOTE oversampling
- **XGBoost** — gradient boosting model
- **LightGBM** — gradient boosting comparison
- **joblib** — model serialization

---

## 📦 Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
**Original source:** Université Libre de Bruxelles (ULB)  
**Size:** 284,807 transactions | 492 fraud cases (0.17%) | 2 days

> The dataset contains only numerical features (V1–V28) resulting 
> from PCA transformation to protect confidentiality.

---

## 🚀 Getting Started

```bash
git clone https://github.com/yumilin92/fraud-detection-analysis.git
cd fraud-detection-analysis

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm joblib

jupyter notebook
```

---

## 👤 Author

**Yulia Vovk**  
Economics background + Data Science  
📍 Tokyo, Japan  
🔗 [Kaggle](https://kaggle.com/yuliavovk) | [GitHub](https://github.com/yumilin92)
