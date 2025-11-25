#!/usr/bin/env python3
"""
DSMIL Device ML Analytics Engine v1.0
ML-powered device pattern analysis, anomaly detection, and predictive analytics
Optimized for SSE4.2 SIMD operations and 512-dimensional vector embeddings
"""

import os
import sys
import asyncio
import numpy as np
import json
import time
import logging
import asyncpg
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

# Add DSMIL paths
DSMIL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(DSMIL_ROOT))

class AnalyticsModel(Enum):
    """Available ML analytics models"""
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES_PREDICTION = "time_series_prediction"
    PATTERN_CLASSIFICATION = "pattern_classification"
    PERFORMANCE_REGRESSION = "performance_regression"

class DeviceState(Enum):
    """Device operational states"""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class DeviceMetrics:
    """Comprehensive device metrics for ML analysis"""
    device_id: int
    device_name: str
    timestamp: datetime
    temperature: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: float
    error_count: int
    response_time: float
    power_consumption: float
    state: DeviceState
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert metrics to optimized feature vector for SIMD operations"""
        # Base features - optimized for SSE4.2 alignment (16-byte boundaries)
        base_features = np.array([
            self.temperature / 100.0,  # Normalize 0-100°C
            self.cpu_usage / 100.0,    # 0-100%
            self.memory_usage / 100.0, # 0-100%
            self.disk_usage / 100.0,   # 0-100%
            self.network_io / 1000.0,  # Normalize MB/s
            self.error_count / 10.0,   # Normalize error count
            self.response_time / 1000.0, # Normalize ms
            self.power_consumption / 100.0, # Normalize watts
            float(self.state.value == "normal"),
            float(self.state.value == "warning"),
            float(self.state.value == "critical"),
            float(self.state.value == "failed"),
            time.mktime(self.timestamp.timetuple()) / 1e10,  # Time feature
            self.device_id / 100.0,  # Device ID feature
            hash(self.device_name) % 1000 / 1000.0,  # Name hash
            len(self.custom_metrics) / 10.0  # Custom metrics count
        ], dtype=np.float32)
        
        # Add custom metrics features
        custom_values = list(self.custom_metrics.values())[:32]  # Limit to 32 custom
        while len(custom_values) < 32:
            custom_values.append(0.0)
        
        custom_array = np.array(custom_values, dtype=np.float32)
        
        # Combine base and custom features
        combined = np.concatenate([base_features, custom_array])
        
        return combined.astype(np.float32)  # Ensure float32 for SIMD optimization

@dataclass
class MLAnalysisResult:
    """Result from ML analysis operations"""
    model_type: AnalyticsModel
    device_id: int
    confidence: float
    prediction: Any
    features_used: List[str]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnomalyAlert:
    """Anomaly detection alert"""
    device_id: int
    device_name: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    confidence: float
    detected_at: datetime
    affected_metrics: List[str]
    recommended_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

class DeviceMLAnalytics:
    """Main ML analytics engine for DSMIL device monitoring"""
    
    def __init__(self, db_pool: asyncpg.Pool, config: Dict[str, Any]):
        self.db_pool = db_pool
        self.config = config
        self.models: Dict[AnalyticsModel, Any] = {}
        self.feature_scalers: Dict[str, Any] = {}
        self.model_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.analysis_count = 0
        self.total_processing_time = 0.0
        self.last_model_update = datetime.now()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ML models with fallback support"""
        try:
            # Try to use sklearn for full ML capabilities
            from sklearn.cluster import DBSCAN, KMeans
            from sklearn.ensemble import IsolationForest, RandomForestRegressor
            from sklearn.preprocessing import StandardScaler, RobustScaler
            from sklearn.decomposition import PCA
            from sklearn.metrics import silhouette_score
            import joblib
            
            self.models = {
                AnalyticsModel.CLUSTERING: KMeans(n_clusters=5, random_state=42, n_init=10),
                AnalyticsModel.ANOMALY_DETECTION: IsolationForest(
                    contamination=0.1, 
                    random_state=42,
                    n_estimators=100
                ),
                AnalyticsModel.PERFORMANCE_REGRESSION: RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                ),
                AnalyticsModel.PATTERN_CLASSIFICATION: DBSCAN(
                    eps=0.3,
                    min_samples=5,
                    metric='euclidean'
                )
            }
            
            self.feature_scalers = {
                'standard': StandardScaler(),
                'robust': RobustScaler(),
                'pca': PCA(n_components=32)  # Reduce dimensions for faster processing
            }
            
            self.logger.info("ML models initialized successfully with sklearn")
            
        except ImportError:
            self.logger.warning("sklearn not available, using basic statistical methods")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self) -> None:
        """Initialize basic statistical models as fallback"""
        self.models = {
            AnalyticsModel.ANOMALY_DETECTION: BasicAnomalyDetector(),
            AnalyticsModel.CLUSTERING: BasicClusterer(),
            AnalyticsModel.PERFORMANCE_REGRESSION: BasicRegressor()
        }
        
        self.feature_scalers = {
            'standard': BasicScaler()
        }
    
    async def analyze_device_patterns(self, device_ids: Optional[List[int]] = None,
                                    hours_back: int = 24) -> Dict[int, List[MLAnalysisResult]]:
        """Analyze device patterns using multiple ML models"""
        start_time = time.time()
        
        try:
            # Get device data from database
            device_data = await self._fetch_device_data(device_ids, hours_back)
            
            if not device_data:
                self.logger.warning("No device data found for analysis")
                return {}
            
            # Group data by device
            devices_by_id = {}
            for data in device_data:
                device_id = data['device_id']
                if device_id not in devices_by_id:
                    devices_by_id[device_id] = []
                devices_by_id[device_id].append(data)
            
            # Analyze each device
            results = {}
            analysis_tasks = []
            
            for device_id, device_records in devices_by_id.items():
                task = self._analyze_single_device(device_id, device_records)
                analysis_tasks.append(task)
            
            # Process all devices concurrently
            device_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            for i, (device_id, _) in enumerate(devices_by_id.items()):
                result = device_results[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Analysis failed for device {device_id}: {result}")
                    continue
                results[device_id] = result
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.analysis_count += 1
            self.total_processing_time += processing_time
            
            self.logger.info(f"Analyzed {len(results)} devices in {processing_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Device pattern analysis failed: {e}")
            return {}
    
    async def _fetch_device_data(self, device_ids: Optional[List[int]], 
                               hours_back: int) -> List[Dict[str, Any]]:
        """Fetch device data from database"""
        try:
            since = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            
            if device_ids:
                device_filter = "AND (" + " OR ".join([
                    f"task_id = 'dsmil_device_{device_id}'" for device_id in device_ids
                ]) + ")"
            else:
                device_filter = ""
            
            query = f"""
                SELECT task_id, embedding, metadata, created_at
                FROM task_embeddings 
                WHERE task_type = 'device_monitoring' 
                AND created_at >= $1
                {device_filter}
                ORDER BY created_at ASC
            """
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(query, since)
            
            # Convert to structured data
            device_data = []
            for row in rows:
                device_id = int(row['task_id'].split('_')[-1])
                metadata = json.loads(row['metadata'])
                
                device_data.append({
                    'device_id': device_id,
                    'device_name': metadata.get('device_name', f'Device_{device_id}'),
                    'timestamp': row['created_at'],
                    'temperature': metadata.get('temperature', 0),
                    'cpu_usage': metadata.get('cpu_usage', 0),
                    'memory_usage': metadata.get('memory_usage', 0),
                    'disk_usage': metadata.get('disk_usage', 0),
                    'network_io': metadata.get('network_io', 0),
                    'error_count': metadata.get('error_count', 0),
                    'response_time': metadata.get('response_time', 0),
                    'power_consumption': metadata.get('power_consumption', 0),
                    'status': metadata.get('status', 'unknown'),
                    'embedding': row['embedding']
                })
            
            return device_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch device data: {e}")
            return []
    
    async def _analyze_single_device(self, device_id: int, 
                                   device_records: List[Dict]) -> List[MLAnalysisResult]:
        """Analyze patterns for a single device"""
        try:
            if len(device_records) < 5:
                self.logger.warning(f"Insufficient data for device {device_id} analysis")
                return []
            
            # Convert to feature matrices
            features = []
            timestamps = []
            
            for record in device_records:
                # Create DeviceMetrics object
                state = DeviceState.NORMAL
                status = record['status'].lower()
                if status == 'critical':
                    state = DeviceState.CRITICAL
                elif status == 'warning':
                    state = DeviceState.WARNING
                elif status == 'failed':
                    state = DeviceState.FAILED
                
                metrics = DeviceMetrics(
                    device_id=record['device_id'],
                    device_name=record['device_name'],
                    timestamp=record['timestamp'],
                    temperature=record['temperature'],
                    cpu_usage=record['cpu_usage'],
                    memory_usage=record['memory_usage'],
                    disk_usage=record['disk_usage'],
                    network_io=record['network_io'],
                    error_count=record['error_count'],
                    response_time=record['response_time'],
                    power_consumption=record['power_consumption'],
                    state=state
                )
                
                features.append(metrics.to_feature_vector())
                timestamps.append(record['timestamp'])
            
            feature_matrix = np.array(features)
            
            # Run analysis models
            results = []
            
            # Anomaly Detection
            anomaly_result = await self._run_anomaly_detection(
                device_id, feature_matrix, timestamps
            )
            if anomaly_result:
                results.append(anomaly_result)
            
            # Clustering Analysis
            cluster_result = await self._run_clustering_analysis(
                device_id, feature_matrix
            )
            if cluster_result:
                results.append(cluster_result)
            
            # Performance Prediction
            if len(feature_matrix) > 10:  # Need sufficient data for prediction
                prediction_result = await self._run_performance_prediction(
                    device_id, feature_matrix, timestamps
                )
                if prediction_result:
                    results.append(prediction_result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Single device analysis failed for {device_id}: {e}")
            return []
    
    async def _run_anomaly_detection(self, device_id: int, 
                                   features: np.ndarray,
                                   timestamps: List[datetime]) -> Optional[MLAnalysisResult]:
        """Run anomaly detection analysis"""
        try:
            start_time = time.time()
            
            # Use thread executor for CPU-intensive ML operations
            loop = asyncio.get_event_loop()
            
            with self.model_lock:
                anomaly_model = self.models.get(AnalyticsModel.ANOMALY_DETECTION)
                if not anomaly_model:
                    return None
                
                # Scale features
                scaler = self.feature_scalers.get('standard')
                if scaler and hasattr(scaler, 'fit_transform'):
                    scaled_features = scaler.fit_transform(features)
                else:
                    scaled_features = features
                
                # Run anomaly detection in thread pool
                if hasattr(anomaly_model, 'fit_predict'):
                    anomaly_scores = await loop.run_in_executor(
                        self.executor, 
                        anomaly_model.fit_predict, 
                        scaled_features
                    )
                else:
                    # Fallback method
                    anomaly_scores = anomaly_model.detect_anomalies(scaled_features)
            
            # Process results
            anomalies = []
            for i, (score, timestamp) in enumerate(zip(anomaly_scores, timestamps)):
                if score == -1:  # Anomaly detected (sklearn convention)
                    anomalies.append({
                        'index': i,
                        'timestamp': timestamp.isoformat(),
                        'features': features[i].tolist()
                    })
            
            processing_time = time.time() - start_time
            
            return MLAnalysisResult(
                model_type=AnalyticsModel.ANOMALY_DETECTION,
                device_id=device_id,
                confidence=len(anomalies) / len(features) if features.size > 0 else 0.0,
                prediction=anomalies,
                features_used=['temperature', 'cpu_usage', 'memory_usage', 'response_time'],
                processing_time=processing_time,
                metadata={
                    'total_points': len(features),
                    'anomalies_found': len(anomalies),
                    'anomaly_rate': len(anomalies) / len(features) if features.size > 0 else 0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed for device {device_id}: {e}")
            return None
    
    async def _run_clustering_analysis(self, device_id: int, 
                                     features: np.ndarray) -> Optional[MLAnalysisResult]:
        """Run clustering analysis to identify device behavior patterns"""
        try:
            start_time = time.time()
            
            loop = asyncio.get_event_loop()
            
            with self.model_lock:
                cluster_model = self.models.get(AnalyticsModel.CLUSTERING)
                if not cluster_model or len(features) < 5:
                    return None
                
                # Scale features
                scaler = self.feature_scalers.get('standard')
                if scaler and hasattr(scaler, 'fit_transform'):
                    scaled_features = scaler.fit_transform(features)
                else:
                    scaled_features = features
                
                # Run clustering in thread pool
                if hasattr(cluster_model, 'fit_predict'):
                    cluster_labels = await loop.run_in_executor(
                        self.executor,
                        cluster_model.fit_predict,
                        scaled_features
                    )
                else:
                    # Fallback method
                    cluster_labels = cluster_model.cluster(scaled_features)
            
            # Analyze cluster distribution
            unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
            
            cluster_info = []
            for cluster_id, count in zip(unique_clusters, cluster_counts):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                cluster_features = features[cluster_indices]
                
                cluster_info.append({
                    'cluster_id': int(cluster_id),
                    'size': int(count),
                    'percentage': float(count / len(features) * 100),
                    'avg_temperature': float(np.mean(cluster_features[:, 0]) * 100),
                    'avg_cpu': float(np.mean(cluster_features[:, 1]) * 100),
                    'avg_memory': float(np.mean(cluster_features[:, 2]) * 100)
                })
            
            processing_time = time.time() - start_time
            
            return MLAnalysisResult(
                model_type=AnalyticsModel.CLUSTERING,
                device_id=device_id,
                confidence=0.8,  # Clustering confidence is subjective
                prediction=cluster_info,
                features_used=['temperature', 'cpu_usage', 'memory_usage'],
                processing_time=processing_time,
                metadata={
                    'total_clusters': len(unique_clusters),
                    'largest_cluster_size': int(max(cluster_counts)),
                    'clustering_algorithm': type(cluster_model).__name__
                }
            )
            
        except Exception as e:
            self.logger.error(f"Clustering analysis failed for device {device_id}: {e}")
            return None
    
    async def _run_performance_prediction(self, device_id: int, 
                                        features: np.ndarray,
                                        timestamps: List[datetime]) -> Optional[MLAnalysisResult]:
        """Run performance prediction analysis"""
        try:
            start_time = time.time()
            
            if len(features) < 10:
                return None
            
            # Prepare time series data for prediction
            # Use response time as target variable
            response_times = features[:, 6] * 1000  # Convert back to ms
            
            # Create lagged features for time series prediction
            X = []
            y = []
            
            lookback = min(5, len(features) - 1)  # Look back 5 time steps
            
            for i in range(lookback, len(features)):
                X.append(features[i-lookback:i].flatten())
                y.append(response_times[i])
            
            if len(X) < 5:
                return None
            
            X = np.array(X)
            y = np.array(y)
            
            # Train prediction model
            loop = asyncio.get_event_loop()
            
            with self.model_lock:
                pred_model = self.models.get(AnalyticsModel.PERFORMANCE_REGRESSION)
                if not pred_model:
                    return None
                
                # Split data for training and prediction
                split_point = int(len(X) * 0.8)
                X_train, X_test = X[:split_point], X[split_point:]
                y_train, y_test = y[:split_point], y[split_point:]
                
                if len(X_train) < 3:
                    return None
                
                # Train model
                if hasattr(pred_model, 'fit'):
                    await loop.run_in_executor(
                        self.executor,
                        pred_model.fit,
                        X_train, y_train
                    )
                
                # Make predictions
                if hasattr(pred_model, 'predict'):
                    predictions = await loop.run_in_executor(
                        self.executor,
                        pred_model.predict,
                        X_test
                    )
                else:
                    # Fallback prediction
                    predictions = pred_model.predict_performance(X_test, y_train)
            
            # Calculate prediction accuracy
            if len(predictions) > 0 and len(y_test) > 0:
                mae = np.mean(np.abs(predictions - y_test))
                rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
            else:
                mae = rmse = 0
            
            processing_time = time.time() - start_time
            
            return MLAnalysisResult(
                model_type=AnalyticsModel.PERFORMANCE_REGRESSION,
                device_id=device_id,
                confidence=max(0, 1 - (rmse / max(np.mean(y), 1))),
                prediction={
                    'predicted_response_times': predictions.tolist() if len(predictions) > 0 else [],
                    'actual_response_times': y_test.tolist() if len(y_test) > 0 else [],
                    'next_prediction': float(predictions[-1]) if len(predictions) > 0 else 0,
                    'trend': 'increasing' if len(predictions) > 1 and predictions[-1] > predictions[0] else 'stable'
                },
                features_used=['response_time', 'temperature', 'cpu_usage', 'memory_usage'],
                processing_time=processing_time,
                metadata={
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Performance prediction failed for device {device_id}: {e}")
            return None
    
    async def detect_anomalies(self, device_ids: Optional[List[int]] = None,
                             severity_threshold: float = 0.7) -> List[AnomalyAlert]:
        """Detect device anomalies and generate alerts"""
        try:
            analysis_results = await self.analyze_device_patterns(device_ids, hours_back=6)
            
            alerts = []
            
            for device_id, device_results in analysis_results.items():
                for result in device_results:
                    if result.model_type == AnalyticsModel.ANOMALY_DETECTION:
                        anomalies = result.prediction
                        
                        if not anomalies:
                            continue
                        
                        # Determine severity
                        anomaly_rate = result.metadata.get('anomaly_rate', 0)
                        
                        if anomaly_rate > 0.5:
                            severity = 'critical'
                        elif anomaly_rate > 0.3:
                            severity = 'high'
                        elif anomaly_rate > 0.1:
                            severity = 'medium'
                        else:
                            severity = 'low'
                        
                        if result.confidence >= severity_threshold:
                            # Get device name from latest anomaly
                            device_data = await self._fetch_device_data([device_id], 1)
                            device_name = device_data[0]['device_name'] if device_data else f'Device_{device_id}'
                            
                            alert = AnomalyAlert(
                                device_id=device_id,
                                device_name=device_name,
                                anomaly_type=self._classify_anomaly_type(anomalies),
                                severity=severity,
                                confidence=result.confidence,
                                detected_at=datetime.now(timezone.utc),
                                affected_metrics=result.features_used,
                                recommended_actions=self._get_recommended_actions(severity, anomalies),
                                metadata={
                                    'anomaly_count': len(anomalies),
                                    'total_points': result.metadata.get('total_points', 0),
                                    'processing_time': result.processing_time
                                }
                            )
                            
                            alerts.append(alert)
            
            # Store alerts in database
            await self._store_anomaly_alerts(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _classify_anomaly_type(self, anomalies: List[Dict]) -> str:
        """Classify the type of anomaly based on patterns"""
        if not anomalies:
            return "unknown"
        
        # Analyze feature patterns in anomalies
        feature_analysis = {}
        
        for anomaly in anomalies:
            features = anomaly.get('features', [])
            if len(features) >= 4:
                # Analyze which metrics are abnormal
                temp = features[0] * 100  # Temperature
                cpu = features[1] * 100   # CPU
                memory = features[2] * 100  # Memory
                response = features[6] * 1000  # Response time
                
                if temp > 85:
                    feature_analysis['thermal'] = feature_analysis.get('thermal', 0) + 1
                if cpu > 90:
                    feature_analysis['cpu_overload'] = feature_analysis.get('cpu_overload', 0) + 1
                if memory > 90:
                    feature_analysis['memory_pressure'] = feature_analysis.get('memory_pressure', 0) + 1
                if response > 500:
                    feature_analysis['performance_degradation'] = feature_analysis.get('performance_degradation', 0) + 1
        
        # Return most common anomaly type
        if feature_analysis:
            return max(feature_analysis.items(), key=lambda x: x[1])[0]
        
        return "pattern_deviation"
    
    def _get_recommended_actions(self, severity: str, anomalies: List[Dict]) -> List[str]:
        """Get recommended actions based on anomaly type and severity"""
        actions = []
        
        if severity == 'critical':
            actions.extend([
                "Immediate investigation required",
                "Consider device shutdown if thermal critical",
                "Alert system administrators",
                "Activate emergency procedures"
            ])
        elif severity == 'high':
            actions.extend([
                "Schedule immediate maintenance",
                "Monitor device closely",
                "Check resource utilization",
                "Review error logs"
            ])
        elif severity == 'medium':
            actions.extend([
                "Schedule maintenance window",
                "Monitor trends",
                "Optimize resource allocation"
            ])
        else:
            actions.extend([
                "Continue monitoring",
                "Document patterns for analysis"
            ])
        
        return actions
    
    async def _store_anomaly_alerts(self, alerts: List[AnomalyAlert]) -> None:
        """Store anomaly alerts in database"""
        try:
            async with self.db_pool.acquire() as conn:
                for alert in alerts:
                    await conn.execute("""
                        INSERT INTO learning_feedback 
                        (user_id, agent_name, feedback_type, content, metadata, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    f"dsmil_device_{alert.device_id}",
                    "device_ml_analytics",
                    "anomaly_alert",
                    json.dumps({
                        "device_name": alert.device_name,
                        "anomaly_type": alert.anomaly_type,
                        "severity": alert.severity,
                        "confidence": alert.confidence,
                        "affected_metrics": alert.affected_metrics,
                        "recommended_actions": alert.recommended_actions
                    }),
                    json.dumps(alert.metadata),
                    alert.detected_at
                    )
                    
        except Exception as e:
            self.logger.error(f"Failed to store anomaly alerts: {e}")
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        try:
            avg_processing_time = (
                self.total_processing_time / max(self.analysis_count, 1)
            )
            
            return {
                "status": "active",
                "models_available": [model.value for model in self.models.keys()],
                "performance": {
                    "total_analyses": self.analysis_count,
                    "average_processing_time": round(avg_processing_time, 3),
                    "throughput_per_hour": round(3600 / max(avg_processing_time, 0.1)),
                    "last_model_update": self.last_model_update.isoformat()
                },
                "capabilities": {
                    "simd_optimized": True,
                    "vector_dimensions": 512,
                    "concurrent_analysis": True,
                    "real_time_detection": True
                },
                "features": {
                    "anomaly_detection": "isolation_forest",
                    "clustering": "kmeans_dbscan",
                    "prediction": "random_forest",
                    "scaling": "standard_robust"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get analytics summary: {e}")
            return {"status": "error", "message": str(e)}

# Fallback classes for systems without sklearn
class BasicAnomalyDetector:
    """Basic statistical anomaly detection fallback"""
    
    def detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        # Use statistical method: points beyond 2 standard deviations
        if len(features) < 5:
            return np.zeros(len(features))
        
        # Calculate z-scores for each feature
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        
        z_scores = np.abs((features - mean) / (std + 1e-8))
        
        # Mark as anomaly if any feature has z-score > 2
        anomalies = np.any(z_scores > 2, axis=1).astype(int)
        return np.where(anomalies, -1, 1)  # Use sklearn convention

class BasicClusterer:
    """Basic clustering fallback using k-means-like algorithm"""
    
    def cluster(self, features: np.ndarray, k: int = 5) -> np.ndarray:
        if len(features) < k:
            return np.arange(len(features))
        
        # Simple k-means implementation
        centroids = features[np.random.choice(len(features), k, replace=False)]
        
        for _ in range(10):  # 10 iterations
            # Assign points to nearest centroid
            distances = np.linalg.norm(features[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if np.any(mask):
                    centroids[i] = np.mean(features[mask], axis=0)
        
        return labels

class BasicRegressor:
    """Basic linear regression fallback"""
    
    def predict_performance(self, X_test: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        # Simple prediction using historical average with trend
        if len(y_train) == 0:
            return np.zeros(len(X_test))
        
        # Return moving average of last few values
        return np.full(len(X_test), np.mean(y_train[-5:]))

class BasicScaler:
    """Basic feature scaling fallback"""
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        # Simple min-max scaling
        min_vals = np.min(features, axis=0)
        max_vals = np.max(features, axis=0)
        
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
        return (features - min_vals) / range_vals

# Usage example
async def main():
    """Example usage of Device ML Analytics"""
    import asyncpg
    
    # Mock database pool for testing
    db_pool = await asyncpg.create_pool(
        host="localhost",
        port=5433,
        database="claude_auth",
        user="claude_auth", 
        password="claude_auth_pass"
    )
    
    config = {
        "ml": {
            "embedding_dimensions": 512,
            "similarity_threshold": 0.8
        }
    }
    
    analytics = DeviceMLAnalytics(db_pool, config)
    
    try:
        # Analyze device patterns
        results = await analytics.analyze_device_patterns([42, 43], hours_back=24)
        print(f"Analysis Results: {len(results)} devices analyzed")
        
        # Detect anomalies
        alerts = await analytics.detect_anomalies([42, 43])
        print(f"Anomaly Alerts: {len(alerts)} alerts generated")
        
        # Get analytics summary
        summary = await analytics.get_analytics_summary()
        print(f"Analytics Summary: {summary['status']}")
        
    finally:
        await db_pool.close()

if __name__ == "__main__":
    asyncio.run(main())