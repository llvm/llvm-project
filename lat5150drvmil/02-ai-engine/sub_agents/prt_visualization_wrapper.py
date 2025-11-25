#!/usr/bin/env python3
"""
PRT-Style Data Visualization & Pattern Recognition Agent
Interactive data exploration, ML visualization, pattern recognition pipelines

Capabilities:
- Interactive dataset exploration and visualization
- Pattern recognition and classification workflows
- Clustering and dimensionality reduction
- ML model visualization and performance analysis
- Feature engineering and selection
- Data preprocessing pipelines

Python-native using: scikit-learn, matplotlib, seaborn, plotly, pandas
Inspired by: MATLAB PRT (Pattern Recognition Toolbox)
"""

import os
import json
from typing import Optional, List, Dict, Any, Union, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PRTVisualizationAgent:
    def __init__(self):
        """Initialize PRT visualization agent"""
        self.data_dir = Path.home() / ".dsmil" / "prt"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets_dir = self.data_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)

        self.visualizations_dir = self.data_dir / "visualizations"
        self.visualizations_dir.mkdir(exist_ok=True)

        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Track loaded datasets and models
        self.datasets = {}
        self.models = {}
        self._load_registries()

        # Check dependencies
        self.dependencies = self._check_dependencies()

    def _check_dependencies(self) -> Dict[str, bool]:
        """Check which libraries are available"""
        deps = {}

        try:
            import pandas
            deps['pandas'] = True
        except ImportError:
            deps['pandas'] = False

        try:
            import numpy
            deps['numpy'] = True
        except ImportError:
            deps['numpy'] = False

        try:
            import sklearn
            deps['sklearn'] = True
        except ImportError:
            deps['sklearn'] = False

        try:
            import matplotlib
            deps['matplotlib'] = True
        except ImportError:
            deps['matplotlib'] = False

        try:
            import seaborn
            deps['seaborn'] = True
        except ImportError:
            deps['seaborn'] = False

        try:
            import plotly
            deps['plotly'] = True
        except ImportError:
            deps['plotly'] = False

        return deps

    def is_available(self) -> bool:
        """Check if agent has minimum required dependencies"""
        # Minimum: pandas, numpy, sklearn
        return all([
            self.dependencies.get('pandas', False),
            self.dependencies.get('numpy', False),
            self.dependencies.get('sklearn', False)
        ])

    def _load_registries(self):
        """Load dataset and model registries"""
        datasets_file = self.data_dir / "datasets.json"
        if datasets_file.exists():
            with open(datasets_file, 'r') as f:
                self.datasets = json.load(f)

        models_file = self.data_dir / "models.json"
        if models_file.exists():
            with open(models_file, 'r') as f:
                self.models = json.load(f)

    def _save_registries(self):
        """Save dataset and model registries"""
        datasets_file = self.data_dir / "datasets.json"
        with open(datasets_file, 'w') as f:
            json.dump(self.datasets, f, indent=2)

        models_file = self.data_dir / "models.json"
        with open(models_file, 'w') as f:
            json.dump(self.models, f, indent=2)

    def load_dataset(self, file_path: str,
                    name: Optional[str] = None,
                    target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Load dataset from file

        Args:
            file_path: Path to CSV/Excel file
            name: Dataset name
            target_column: Target/label column for supervised learning

        Returns:
            Dict with dataset info
        """
        if not self.dependencies.get('pandas'):
            return {
                "success": False,
                "error": "Pandas not installed. Install: pip install pandas"
            }

        import pandas as pd
        import numpy as np

        file_path = Path(file_path)
        if not file_path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        try:
            # Load based on extension
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file format: {file_path.suffix}"
                }

            # Generate dataset ID
            dataset_id = f"ds_{len(self.datasets) + 1}"

            if not name:
                name = file_path.stem

            # Analyze dataset
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Save dataset
            dataset_file = self.datasets_dir / f"{dataset_id}.csv"
            df.to_csv(dataset_file, index=False)

            # Register dataset
            self.datasets[dataset_id] = {
                "id": dataset_id,
                "name": name,
                "file_path": str(file_path),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "target_column": target_column,
                "missing_values": df.isnull().sum().sum(),
                "saved_path": str(dataset_file)
            }

            self._save_registries()

            return {
                "success": True,
                "dataset_id": dataset_id,
                "name": name,
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load dataset: {str(e)}"
            }

    def explore_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate exploratory data analysis

        Args:
            dataset_id: Dataset ID

        Returns:
            Dict with statistical summary
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        import pandas as pd

        dataset_file = Path(self.datasets[dataset_id]["saved_path"])
        df = pd.read_csv(dataset_file)

        # Statistical summary
        summary = {
            "basic_stats": df.describe().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns}
        }

        return {
            "success": True,
            "dataset_id": dataset_id,
            "summary": summary
        }

    def visualize_dataset(self, dataset_id: str,
                         viz_type: str = "correlation",
                         columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create dataset visualizations

        Args:
            dataset_id: Dataset ID
            viz_type: 'correlation', 'distribution', 'scatter', 'pairplot', 'box'
            columns: Specific columns to visualize (None = all numeric)

        Returns:
            Dict with visualization file path
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        import pandas as pd
        import numpy as np

        dataset_file = Path(self.datasets[dataset_id]["saved_path"])
        df = pd.read_csv(dataset_file)

        # Select columns
        if columns is None:
            columns = self.datasets[dataset_id]["numeric_columns"]

        if not columns:
            return {
                "success": False,
                "error": "No numeric columns available for visualization"
            }

        try:
            if viz_type == "correlation":
                result = self._viz_correlation(df[columns], dataset_id)
            elif viz_type == "distribution":
                result = self._viz_distribution(df[columns], dataset_id)
            elif viz_type == "scatter":
                result = self._viz_scatter(df[columns], dataset_id)
            elif viz_type == "pairplot":
                result = self._viz_pairplot(df[columns], dataset_id)
            elif viz_type == "box":
                result = self._viz_boxplot(df[columns], dataset_id)
            else:
                return {
                    "success": False,
                    "error": f"Unknown visualization type: {viz_type}"
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Visualization failed: {str(e)}"
            }

    def _viz_correlation(self, df, dataset_id: str) -> Dict[str, Any]:
        """Create correlation heatmap"""
        if not self.dependencies.get('matplotlib') or not self.dependencies.get('seaborn'):
            return {
                "success": False,
                "error": "Matplotlib and Seaborn required for correlation plot"
            }

        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')

        output_file = self.visualizations_dir / f"{dataset_id}_correlation.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "success": True,
            "viz_type": "correlation",
            "file_path": str(output_file)
        }

    def _viz_distribution(self, df, dataset_id: str) -> Dict[str, Any]:
        """Create distribution plots"""
        if not self.dependencies.get('matplotlib'):
            return {
                "success": False,
                "error": "Matplotlib required for distribution plots"
            }

        import matplotlib.pyplot as plt

        n_cols = len(df.columns)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_cols > 1 else [axes]

        for idx, col in enumerate(df.columns):
            axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[idx].set_title(f'Distribution: {col}')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('Frequency')

        # Hide unused subplots
        for idx in range(n_cols, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        output_file = self.visualizations_dir / f"{dataset_id}_distribution.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "success": True,
            "viz_type": "distribution",
            "file_path": str(output_file)
        }

    def _viz_scatter(self, df, dataset_id: str) -> Dict[str, Any]:
        """Create scatter plot matrix"""
        if not self.dependencies.get('plotly'):
            return {
                "success": False,
                "error": "Plotly required for interactive scatter plots"
            }

        import plotly.express as px

        # Use first two columns for scatter
        if len(df.columns) < 2:
            return {
                "success": False,
                "error": "Need at least 2 numeric columns for scatter plot"
            }

        fig = px.scatter(df, x=df.columns[0], y=df.columns[1],
                        title=f'Scatter: {df.columns[0]} vs {df.columns[1]}')

        output_file = self.visualizations_dir / f"{dataset_id}_scatter.html"
        fig.write_html(str(output_file))

        return {
            "success": True,
            "viz_type": "scatter",
            "file_path": str(output_file)
        }

    def _viz_pairplot(self, df, dataset_id: str) -> Dict[str, Any]:
        """Create pairplot"""
        if not self.dependencies.get('seaborn'):
            return {
                "success": False,
                "error": "Seaborn required for pairplot"
            }

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Limit to first 5 columns for performance
        plot_df = df.iloc[:, :5]

        sns.pairplot(plot_df)

        output_file = self.visualizations_dir / f"{dataset_id}_pairplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "success": True,
            "viz_type": "pairplot",
            "file_path": str(output_file)
        }

    def _viz_boxplot(self, df, dataset_id: str) -> Dict[str, Any]:
        """Create box plots"""
        if not self.dependencies.get('matplotlib'):
            return {
                "success": False,
                "error": "Matplotlib required for box plots"
            }

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        df.boxplot()
        plt.title('Box Plots - Outlier Detection')
        plt.xticks(rotation=45)

        output_file = self.visualizations_dir / f"{dataset_id}_boxplot.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "success": True,
            "viz_type": "boxplot",
            "file_path": str(output_file)
        }

    def train_classifier(self, dataset_id: str,
                        algorithm: str = "random_forest",
                        test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train classification model

        Args:
            dataset_id: Dataset ID
            algorithm: 'random_forest', 'svm', 'logistic', 'knn', 'decision_tree'
            test_size: Test set proportion

        Returns:
            Dict with model performance
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        if not self.datasets[dataset_id].get('target_column'):
            return {
                "success": False,
                "error": "No target column specified. Set target_column when loading dataset."
            }

        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        # Load dataset
        dataset_file = Path(self.datasets[dataset_id]["saved_path"])
        df = pd.read_csv(dataset_file)

        target_col = self.datasets[dataset_id]['target_column']

        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model based on algorithm
        try:
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif algorithm == "svm":
                from sklearn.svm import SVC
                model = SVC(kernel='rbf', random_state=42)
            elif algorithm == "logistic":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(random_state=42, max_iter=1000)
            elif algorithm == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors=5)
            elif algorithm == "decision_tree":
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(random_state=42)
            else:
                return {
                    "success": False,
                    "error": f"Unknown algorithm: {algorithm}"
                }

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred, output_dict=True)

            # Save model
            import joblib
            model_id = f"model_{len(self.models) + 1}"
            model_file = self.models_dir / f"{model_id}.joblib"
            joblib.dump({
                'model': model,
                'scaler': scaler,
                'feature_names': X.columns.tolist()
            }, model_file)

            # Register model
            self.models[model_id] = {
                "id": model_id,
                "dataset_id": dataset_id,
                "algorithm": algorithm,
                "accuracy": accuracy,
                "file_path": str(model_file)
            }

            self._save_registries()

            return {
                "success": True,
                "model_id": model_id,
                "algorithm": algorithm,
                "accuracy": round(accuracy, 4),
                "confusion_matrix": conf_matrix,
                "classification_report": class_report
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Training failed: {str(e)}"
            }

    def cluster_analysis(self, dataset_id: str,
                        n_clusters: int = 3,
                        algorithm: str = "kmeans") -> Dict[str, Any]:
        """
        Perform clustering analysis

        Args:
            dataset_id: Dataset ID
            n_clusters: Number of clusters
            algorithm: 'kmeans', 'dbscan', 'hierarchical'

        Returns:
            Dict with clustering results
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Load dataset
        dataset_file = Path(self.datasets[dataset_id]["saved_path"])
        df = pd.read_csv(dataset_file)

        # Use numeric columns
        numeric_cols = self.datasets[dataset_id]["numeric_columns"]
        X = df[numeric_cols].dropna()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            if algorithm == "kmeans":
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            elif algorithm == "dbscan":
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(eps=0.5, min_samples=5)
            elif algorithm == "hierarchical":
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                return {
                    "success": False,
                    "error": f"Unknown algorithm: {algorithm}"
                }

            # Perform clustering
            labels = clusterer.fit_predict(X_scaled)

            # Calculate metrics
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0

            return {
                "success": True,
                "algorithm": algorithm,
                "n_clusters": len(set(labels)),
                "silhouette_score": round(silhouette, 4),
                "cluster_sizes": {int(label): int((labels == label).sum()) for label in set(labels)}
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Clustering failed: {str(e)}"
            }

    def dimensionality_reduction(self, dataset_id: str,
                                method: str = "pca",
                                n_components: int = 2) -> Dict[str, Any]:
        """
        Perform dimensionality reduction

        Args:
            dataset_id: Dataset ID
            method: 'pca', 'tsne', 'umap'
            n_components: Number of components

        Returns:
            Dict with reduction results and visualization
        """
        if dataset_id not in self.datasets:
            return {
                "success": False,
                "error": f"Dataset {dataset_id} not found"
            }

        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler

        # Load dataset
        dataset_file = Path(self.datasets[dataset_id]["saved_path"])
        df = pd.read_csv(dataset_file)

        # Use numeric columns
        numeric_cols = self.datasets[dataset_id]["numeric_columns"]
        X = df[numeric_cols].dropna()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            if method == "pca":
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components)
                explained_var = None
            elif method == "tsne":
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
            else:
                return {
                    "success": False,
                    "error": f"Unknown method: {method}. Try 'pca' or 'tsne'"
                }

            # Perform reduction
            X_reduced = reducer.fit_transform(X_scaled)

            # Get explained variance for PCA
            if method == "pca":
                explained_var = reducer.explained_variance_ratio_.tolist()

            # Visualize if 2D
            viz_file = None
            if n_components == 2 and self.dependencies.get('matplotlib'):
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 8))
                plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
                plt.title(f'{method.upper()} - 2D Projection')

                viz_file = self.visualizations_dir / f"{dataset_id}_{method}.png"
                plt.savefig(viz_file, dpi=300, bbox_inches='tight')
                plt.close()

            return {
                "success": True,
                "method": method,
                "n_components": n_components,
                "explained_variance": explained_var,
                "visualization": str(viz_file) if viz_file else None
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Dimensionality reduction failed: {str(e)}"
            }

    def list_datasets(self) -> Dict[str, Any]:
        """List all datasets"""
        return {
            "success": True,
            "datasets": list(self.datasets.values()),
            "count": len(self.datasets)
        }

    def list_models(self) -> Dict[str, Any]:
        """List all trained models"""
        return {
            "success": True,
            "models": list(self.models.values()),
            "count": len(self.models)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "available": self.is_available(),
            "dependencies": self.dependencies,
            "datasets_loaded": len(self.datasets),
            "models_trained": len(self.models),
            "visualizations_created": len(list(self.visualizations_dir.glob('*.*'))),
            "storage_path": str(self.data_dir)
        }

# Export
__all__ = ['PRTVisualizationAgent']
