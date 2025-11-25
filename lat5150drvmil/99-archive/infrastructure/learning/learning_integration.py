#!/usr/bin/env python3
"""
DSMIL Learning Integration System v1.0
Connects DSMIL device monitoring to PostgreSQL Enhanced Learning System
Provides ML-powered device pattern analysis and agent coordination
"""

import os
import sys
import asyncio
import asyncpg
import json
import time
import logging
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import subprocess

# Add DSMIL paths
DSMIL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(DSMIL_ROOT / "01-source" / "monitor"))
sys.path.insert(0, str(DSMIL_ROOT))

class DSMILLearningStatus(Enum):
    """Status levels for DSMIL learning integration"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    OFFLINE = "offline"
    ERROR = "error"

@dataclass
class DevicePattern:
    """Device pattern data for ML analysis"""
    device_id: int
    device_name: str
    temperature: float
    cpu_usage: float
    memory_usage: float
    error_count: int
    response_time: float
    status: str
    timestamp: datetime
    vector_embedding: Optional[np.ndarray] = None
    
    def to_vector(self) -> np.ndarray:
        """Convert device metrics to 512-dimensional vector for ML analysis"""
        base_features = np.array([
            self.device_id / 100.0,  # Normalize device ID
            self.temperature / 100.0,  # Normalize temperature (0-100°C)
            self.cpu_usage / 100.0,  # CPU usage percentage
            self.memory_usage / 100.0,  # Memory usage percentage
            self.error_count / 10.0,  # Normalize error count
            self.response_time / 1000.0,  # Normalize response time (ms)
            hash(self.status) % 100 / 100.0,  # Status hash
            time.mktime(self.timestamp.timetuple()) / 1e10  # Timestamp
        ])
        
        # Expand to 512 dimensions with engineered features
        extended_features = []
        
        # Add polynomial features
        for i in range(len(base_features)):
            for j in range(i, len(base_features)):
                extended_features.append(base_features[i] * base_features[j])
        
        # Add sine/cosine transforms for cyclical patterns
        for val in base_features:
            extended_features.extend([np.sin(2 * np.pi * val), np.cos(2 * np.pi * val)])
        
        # Add statistical moments
        extended_features.extend([
            np.mean(base_features),
            np.std(base_features),
            np.min(base_features),
            np.max(base_features)
        ])
        
        # Pad or truncate to exactly 512 dimensions
        extended_array = np.array(extended_features)
        if len(extended_array) < 512:
            padding = np.zeros(512 - len(extended_array))
            extended_array = np.concatenate([extended_array, padding])
        elif len(extended_array) > 512:
            extended_array = extended_array[:512]
            
        self.vector_embedding = extended_array
        return extended_array

@dataclass
class AgentRecommendation:
    """ML-powered agent recommendation"""
    agent_name: str
    confidence: float
    task_type: str
    estimated_duration: float
    required_resources: Dict[str, Any]
    priority: int

class DSMILLearningIntegrator:
    """Main integration class connecting DSMIL to Enhanced Learning System"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DSMIL_ROOT / "infrastructure" / "learning" / "config.json"
        self.db_pool: Optional[asyncpg.Pool] = None
        self.status = DSMILLearningStatus.INITIALIZING
        self.device_patterns: Dict[int, DevicePattern] = {}
        self.ml_models: Dict[str, Any] = {}
        self.agent_performance_cache: Dict[str, float] = {}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create defaults"""
        default_config = {
            "database": {
                "host": "localhost",
                "port": 5433,
                "database": "claude_auth",
                "user": "claude_auth",
                "password": "claude_auth_pass",
                "min_connections": 5,
                "max_connections": 20
            },
            "dsmil": {
                "monitor_interval": 1.0,
                "thermal_warning": 75,
                "thermal_critical": 85,
                "device_timeout": 5.0
            },
            "ml": {
                "embedding_dimensions": 512,
                "similarity_threshold": 0.8,
                "batch_size": 100,
                "learning_rate": 0.001
            },
            "agents": {
                "max_concurrent": 10,
                "default_timeout": 30.0,
                "retry_attempts": 3
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}, using defaults")
                
        return default_config
    
    async def initialize(self) -> bool:
        """Initialize the learning integration system"""
        try:
            self.logger.info("Initializing DSMIL Learning Integration System...")
            
            # Initialize database connection pool
            db_config = self.config["database"]
            self.db_pool = await asyncpg.create_pool(
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                user=db_config["user"],
                password=db_config["password"],
                min_size=db_config["min_connections"],
                max_size=db_config["max_connections"]
            )
            
            # Verify database tables exist
            await self._verify_database_schema()
            
            # Initialize ML models
            await self._initialize_ml_models()
            
            # Load agent performance cache
            await self._load_agent_performance()
            
            self.status = DSMILLearningStatus.ACTIVE
            self.logger.info("DSMIL Learning Integration System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            self.status = DSMILLearningStatus.ERROR
            return False
    
    async def _verify_database_schema(self) -> bool:
        """Verify required database tables exist"""
        required_tables = [
            "agent_metrics",
            "task_embeddings", 
            "learning_feedback",
            "model_performance",
            "interaction_logs"
        ]
        
        async with self.db_pool.acquire() as conn:
            for table in required_tables:
                result = await conn.fetchval(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = $1)",
                    table
                )
                if not result:
                    self.logger.error(f"Required table {table} not found in database")
                    return False
        
        self.logger.info("Database schema verification complete")
        return True
    
    async def _initialize_ml_models(self) -> None:
        """Initialize ML models for pattern analysis"""
        try:
            # Try to import sklearn for ML capabilities
            from sklearn.cluster import KMeans
            from sklearn.ensemble import IsolationForest
            from sklearn.preprocessing import StandardScaler
            
            self.ml_models = {
                "device_clustering": KMeans(n_clusters=5, random_state=42),
                "anomaly_detection": IsolationForest(contamination=0.1, random_state=42),
                "feature_scaler": StandardScaler()
            }
            
            self.logger.info("ML models initialized successfully")
            
        except ImportError:
            self.logger.warning("sklearn not available, using basic ML fallbacks")
            self.ml_models = {}
    
    async def _load_agent_performance(self) -> None:
        """Load agent performance data from database"""
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT agent_name, AVG(execution_time) as avg_time, 
                           AVG(success_rate) as avg_success
                    FROM agent_metrics 
                    WHERE created_at > NOW() - INTERVAL '7 days'
                    GROUP BY agent_name
                """)
                
                for row in rows:
                    # Composite performance score
                    performance = (row['avg_success'] * 0.7) + ((1000 / max(row['avg_time'], 1)) * 0.3)
                    self.agent_performance_cache[row['agent_name']] = performance
                    
                self.logger.info(f"Loaded performance data for {len(rows)} agents")
                
        except Exception as e:
            self.logger.warning(f"Failed to load agent performance: {e}")
    
    async def record_device_pattern(self, device_data: Dict[str, Any]) -> bool:
        """Record device pattern and store in learning database"""
        try:
            pattern = DevicePattern(
                device_id=device_data.get('device_id', 0),
                device_name=device_data.get('name', 'unknown'),
                temperature=float(device_data.get('temperature', 0)),
                cpu_usage=float(device_data.get('cpu_usage', 0)),
                memory_usage=float(device_data.get('memory_usage', 0)),
                error_count=int(device_data.get('errors', 0)),
                response_time=float(device_data.get('response_time', 0)),
                status=device_data.get('status', 'unknown'),
                timestamp=datetime.now(timezone.utc)
            )
            
            # Generate vector embedding
            vector = pattern.to_vector()
            
            # Store in database
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO task_embeddings 
                    (task_id, task_type, embedding, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                """, 
                f"dsmil_device_{pattern.device_id}",
                "device_monitoring",
                vector.tolist(),
                json.dumps({
                    "device_name": pattern.device_name,
                    "temperature": pattern.temperature,
                    "cpu_usage": pattern.cpu_usage,
                    "memory_usage": pattern.memory_usage,
                    "error_count": pattern.error_count,
                    "response_time": pattern.response_time,
                    "status": pattern.status
                }),
                pattern.timestamp
                )
            
            # Update local cache
            self.device_patterns[pattern.device_id] = pattern
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record device pattern: {e}")
            return False
    
    async def analyze_device_anomalies(self, device_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Analyze device patterns for anomalies using ML"""
        try:
            # Get recent device patterns
            async with self.db_pool.acquire() as conn:
                if device_id:
                    rows = await conn.fetch("""
                        SELECT * FROM task_embeddings 
                        WHERE task_id = $1 AND task_type = 'device_monitoring'
                        ORDER BY created_at DESC LIMIT 100
                    """, f"dsmil_device_{device_id}")
                else:
                    rows = await conn.fetch("""
                        SELECT * FROM task_embeddings 
                        WHERE task_type = 'device_monitoring'
                        ORDER BY created_at DESC LIMIT 1000
                    """)
            
            if not rows or not self.ml_models.get("anomaly_detection"):
                return []
            
            # Extract embeddings and metadata
            embeddings = []
            metadata = []
            
            for row in rows:
                embeddings.append(np.array(row['embedding']))
                metadata.append({
                    'task_id': row['task_id'],
                    'timestamp': row['created_at'],
                    'metadata': json.loads(row['metadata'])
                })
            
            embeddings_array = np.array(embeddings)
            
            # Detect anomalies
            anomaly_detector = self.ml_models["anomaly_detection"]
            anomaly_scores = anomaly_detector.fit_predict(embeddings_array)
            
            # Collect anomalies
            anomalies = []
            for i, (score, meta) in enumerate(zip(anomaly_scores, metadata)):
                if score == -1:  # Anomaly detected
                    anomalies.append({
                        'device_id': meta['task_id'].split('_')[-1],
                        'timestamp': meta['timestamp'],
                        'severity': 'high' if meta['metadata'].get('error_count', 0) > 5 else 'medium',
                        'details': meta['metadata'],
                        'anomaly_type': self._classify_anomaly(meta['metadata'])
                    })
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to analyze anomalies: {e}")
            return []
    
    def _classify_anomaly(self, metadata: Dict[str, Any]) -> str:
        """Classify type of anomaly based on device metrics"""
        temp = metadata.get('temperature', 0)
        cpu = metadata.get('cpu_usage', 0)
        mem = metadata.get('memory_usage', 0)
        errors = metadata.get('error_count', 0)
        
        if temp > 90:
            return "thermal_critical"
        elif cpu > 95:
            return "cpu_overload"
        elif mem > 95:
            return "memory_exhaustion"
        elif errors > 10:
            return "error_burst"
        elif temp > 75 and cpu > 80:
            return "performance_degradation"
        else:
            return "pattern_deviation"
    
    async def recommend_agents(self, task_description: str, device_context: Optional[Dict] = None) -> List[AgentRecommendation]:
        """Get ML-powered agent recommendations for a task"""
        try:
            # Generate task embedding
            task_vector = self._generate_task_embedding(task_description, device_context)
            
            # Find similar tasks in database
            similar_tasks = await self._find_similar_tasks(task_vector)
            
            # Get agent recommendations based on similar tasks and performance
            recommendations = []
            
            # Agent specialization mapping
            agent_specializations = {
                "hardware": ["temperature", "thermal", "cpu", "memory", "device"],
                "security": ["security", "audit", "breach", "attack", "vulnerability"],
                "monitor": ["monitoring", "alerting", "tracking", "surveillance"],
                "optimizer": ["performance", "optimization", "efficiency", "speed"],
                "debugger": ["error", "debug", "troubleshoot", "fault", "failure"],
                "database": ["data", "database", "storage", "query", "analytics"],
                "infrastructure": ["deployment", "infrastructure", "system", "network"],
                "mlops": ["machine learning", "model", "prediction", "analysis"]
            }
            
            # Score agents based on task relevance
            task_lower = task_description.lower()
            for agent_name, keywords in agent_specializations.items():
                relevance_score = sum(1 for keyword in keywords if keyword in task_lower)
                if relevance_score > 0:
                    performance_score = self.agent_performance_cache.get(agent_name, 0.5)
                    combined_score = (relevance_score * 0.6) + (performance_score * 0.4)
                    
                    recommendations.append(AgentRecommendation(
                        agent_name=agent_name,
                        confidence=min(combined_score, 1.0),
                        task_type=self._classify_task_type(task_description),
                        estimated_duration=self._estimate_duration(task_description, agent_name),
                        required_resources=self._estimate_resources(agent_name, device_context),
                        priority=int(combined_score * 10)
                    ))
            
            # Sort by confidence score
            recommendations.sort(key=lambda x: x.confidence, reverse=True)
            
            return recommendations[:5]  # Return top 5 recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to recommend agents: {e}")
            return []
    
    def _generate_task_embedding(self, task: str, context: Optional[Dict] = None) -> np.ndarray:
        """Generate 512-dimensional embedding for task description"""
        # Simple word-based embedding (in production, use proper NLP models)
        words = task.lower().split()
        word_hashes = [hash(word) % 1000 / 1000.0 for word in words]
        
        # Add context features if available
        context_features = []
        if context:
            context_features = [
                context.get('temperature', 0) / 100.0,
                context.get('cpu_usage', 0) / 100.0,
                context.get('memory_usage', 0) / 100.0,
                context.get('error_count', 0) / 10.0
            ]
        
        # Combine features
        combined_features = word_hashes + context_features
        
        # Pad or truncate to 512 dimensions
        if len(combined_features) < 512:
            padding = np.zeros(512 - len(combined_features))
            combined_features.extend(padding)
        else:
            combined_features = combined_features[:512]
            
        return np.array(combined_features)
    
    async def _find_similar_tasks(self, query_vector: np.ndarray, limit: int = 10) -> List[Dict]:
        """Find similar tasks using vector similarity search"""
        try:
            async with self.db_pool.acquire() as conn:
                # Use cosine similarity for vector search
                # Note: This is a simplified version, production should use pgvector operators
                rows = await conn.fetch("""
                    SELECT task_id, task_type, embedding, metadata, created_at
                    FROM task_embeddings
                    ORDER BY created_at DESC
                    LIMIT 1000
                """)
                
                similarities = []
                for row in rows:
                    stored_vector = np.array(row['embedding'])
                    # Cosine similarity
                    similarity = np.dot(query_vector, stored_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
                    )
                    
                    if similarity > self.config["ml"]["similarity_threshold"]:
                        similarities.append({
                            'task_id': row['task_id'],
                            'task_type': row['task_type'],
                            'similarity': float(similarity),
                            'metadata': json.loads(row['metadata']),
                            'created_at': row['created_at']
                        })
                
                # Sort by similarity
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                return similarities[:limit]
                
        except Exception as e:
            self.logger.error(f"Failed to find similar tasks: {e}")
            return []
    
    def _classify_task_type(self, task: str) -> str:
        """Classify task type based on description"""
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["monitor", "watch", "track"]):
            return "monitoring"
        elif any(word in task_lower for word in ["fix", "repair", "debug", "troubleshoot"]):
            return "troubleshooting"
        elif any(word in task_lower for word in ["optimize", "improve", "enhance"]):
            return "optimization"
        elif any(word in task_lower for word in ["analyze", "investigate", "examine"]):
            return "analysis"
        elif any(word in task_lower for word in ["deploy", "install", "setup"]):
            return "deployment"
        else:
            return "general"
    
    def _estimate_duration(self, task: str, agent: str) -> float:
        """Estimate task duration in minutes"""
        base_durations = {
            "hardware": 15.0,
            "security": 30.0,
            "monitor": 5.0,
            "optimizer": 20.0,
            "debugger": 25.0,
            "database": 10.0,
            "infrastructure": 45.0,
            "mlops": 35.0
        }
        
        complexity_multiplier = 1.0
        if any(word in task.lower() for word in ["complex", "comprehensive", "full", "complete"]):
            complexity_multiplier = 2.0
        elif any(word in task.lower() for word in ["quick", "simple", "basic"]):
            complexity_multiplier = 0.5
            
        return base_durations.get(agent, 20.0) * complexity_multiplier
    
    def _estimate_resources(self, agent: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Estimate required resources for agent execution"""
        base_resources = {
            "hardware": {"cpu": 2, "memory": "1GB", "disk": "100MB"},
            "security": {"cpu": 1, "memory": "512MB", "disk": "50MB"},
            "monitor": {"cpu": 1, "memory": "256MB", "disk": "10MB"},
            "optimizer": {"cpu": 4, "memory": "2GB", "disk": "200MB"},
            "debugger": {"cpu": 2, "memory": "1GB", "disk": "500MB"},
            "database": {"cpu": 2, "memory": "1GB", "disk": "1GB"},
            "infrastructure": {"cpu": 2, "memory": "2GB", "disk": "1GB"},
            "mlops": {"cpu": 4, "memory": "4GB", "disk": "2GB"}
        }
        
        resources = base_resources.get(agent, {"cpu": 1, "memory": "512MB", "disk": "100MB"})
        
        # Adjust based on context
        if context and context.get('error_count', 0) > 5:
            resources["cpu"] = resources["cpu"] * 2
            
        return resources
    
    async def record_agent_interaction(self, agent_name: str, task: str, 
                                     duration: float, success: bool, 
                                     result_metadata: Optional[Dict] = None) -> bool:
        """Record agent interaction for learning"""
        try:
            async with self.db_pool.acquire() as conn:
                # Record in agent_metrics
                await conn.execute("""
                    INSERT INTO agent_metrics 
                    (agent_name, task_type, execution_time, success_rate, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """,
                agent_name,
                self._classify_task_type(task),
                duration,
                1.0 if success else 0.0,
                json.dumps(result_metadata or {}),
                datetime.now(timezone.utc)
                )
                
                # Record in interaction_logs
                await conn.execute("""
                    INSERT INTO interaction_logs 
                    (source_agent, target_agent, interaction_type, payload, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                """,
                "dsmil_integrator",
                agent_name,
                "task_execution",
                json.dumps({"task": task, "duration": duration, "success": success}),
                datetime.now(timezone.utc)
                )
                
            # Update performance cache
            current_perf = self.agent_performance_cache.get(agent_name, 0.5)
            new_perf = (current_perf * 0.8) + ((1.0 if success else 0.0) * 0.2)
            self.agent_performance_cache[agent_name] = new_perf
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to record agent interaction: {e}")
            return False
    
    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary with ML insights"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get recent device metrics
                device_health = await conn.fetch("""
                    SELECT COUNT(*) as total_devices,
                           AVG(CAST(metadata->>'temperature' AS FLOAT)) as avg_temp,
                           SUM(CAST(metadata->>'error_count' AS INT)) as total_errors
                    FROM task_embeddings 
                    WHERE task_type = 'device_monitoring' 
                    AND created_at > NOW() - INTERVAL '1 hour'
                """)
                
                # Get agent performance
                agent_performance = await conn.fetch("""
                    SELECT agent_name, COUNT(*) as tasks,
                           AVG(execution_time) as avg_time,
                           AVG(success_rate) as success_rate
                    FROM agent_metrics 
                    WHERE created_at > NOW() - INTERVAL '24 hours'
                    GROUP BY agent_name
                    ORDER BY success_rate DESC
                """)
                
                # Detect anomalies
                anomalies = await self.analyze_device_anomalies()
                
                return {
                    "status": self.status.value,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "device_health": {
                        "total_devices": device_health[0]['total_devices'] if device_health else 0,
                        "average_temperature": round(device_health[0]['avg_temp'] or 0, 2),
                        "total_errors": device_health[0]['total_errors'] if device_health else 0
                    },
                    "agent_performance": [
                        {
                            "name": row['agent_name'],
                            "tasks_completed": row['tasks'],
                            "average_duration": round(row['avg_time'], 2),
                            "success_rate": round(row['success_rate'] * 100, 1)
                        }
                        for row in agent_performance
                    ],
                    "anomalies": {
                        "count": len(anomalies),
                        "critical": len([a for a in anomalies if a['severity'] == 'high']),
                        "recent": anomalies[:5]
                    },
                    "ml_models": {
                        "available": list(self.ml_models.keys()),
                        "embedding_dimensions": self.config["ml"]["embedding_dimensions"]
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get system health summary: {e}")
            return {"status": "error", "message": str(e)}
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the integration system"""
        self.logger.info("Shutting down DSMIL Learning Integration System...")
        
        if self.db_pool:
            await self.db_pool.close()
            
        self.status = DSMILLearningStatus.OFFLINE
        self.logger.info("Shutdown complete")

# Usage example
async def main():
    """Example usage of DSMIL Learning Integration"""
    integrator = DSMILLearningIntegrator()
    
    try:
        # Initialize system
        if not await integrator.initialize():
            print("Failed to initialize integration system")
            return
        
        # Example device data
        device_data = {
            "device_id": 42,
            "name": "DSMIL_Thermal_Sensor",
            "temperature": 78.5,
            "cpu_usage": 65.2,
            "memory_usage": 45.8,
            "errors": 2,
            "response_time": 125.5,
            "status": "active"
        }
        
        # Record device pattern
        await integrator.record_device_pattern(device_data)
        
        # Get agent recommendations
        recommendations = await integrator.recommend_agents(
            "Monitor thermal conditions and optimize performance",
            device_data
        )
        
        print("Agent Recommendations:")
        for rec in recommendations:
            print(f"  {rec.agent_name}: {rec.confidence:.2f} confidence")
        
        # Analyze anomalies
        anomalies = await integrator.analyze_device_anomalies()
        if anomalies:
            print(f"\nDetected {len(anomalies)} anomalies")
        
        # Get system health
        health = await integrator.get_system_health_summary()
        print(f"\nSystem Health: {health['status']}")
        print(f"Total Devices: {health.get('device_health', {}).get('total_devices', 0)}")
        
    finally:
        await integrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())