#!/usr/bin/env python3
"""
DSMIL ML Orchestrator v1.0
Main orchestrator that integrates all ML-powered DSMIL components:
- Learning Integration (PostgreSQL connection)
- Device ML Analytics (pattern analysis and anomaly detection)  
- Agent Coordinator (80-agent orchestration with ML selection)

Optimized for Intel Meteor Lake with SSE4.2 SIMD operations
"""

import os
import sys
import asyncio
import asyncpg
import json
import time
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import argparse

# Add local modules
DSMIL_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DSMIL_ROOT / "infrastructure" / "learning"))
sys.path.insert(0, str(DSMIL_ROOT / "infrastructure" / "coordination"))

try:
    from learning_integration import DSMILLearningIntegrator, DevicePattern
    from device_ml_analytics import DeviceMLAnalytics, AnalyticsModel, AnomalyAlert
    from agent_coordinator import DSMILAgentCoordinator, TaskRequest, TaskPriority, ExecutionResult
except ImportError as e:
    print(f"Failed to import DSMIL modules: {e}")
    print("Ensure all required modules are in the correct paths")
    sys.exit(1)

class OrchestrationMode:
    """Orchestration modes for different use cases"""
    MONITORING = "monitoring"  # Pure monitoring and data collection
    ANALYSIS = "analysis"      # ML analysis and pattern detection
    COORDINATION = "coordination"  # Full agent coordination
    INTEGRATED = "integrated"  # All systems integrated

@dataclass
class SystemHealth:
    """Overall system health status"""
    learning_system_status: str
    ml_analytics_status: str
    agent_coordinator_status: str
    database_connection: bool
    total_devices_monitored: int
    active_agents: int
    recent_success_rate: float
    avg_response_time: float
    anomalies_detected: int
    last_update: datetime

class DSMILMLOrchestrator:
    """Main orchestrator for DSMIL ML-powered monitoring and agent coordination"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or DSMIL_ROOT / "infrastructure" / "learning" / "config.json"
        self.config = self._load_config()
        
        # Core components
        self.learning_integrator: Optional[DSMILLearningIntegrator] = None
        self.ml_analytics: Optional[DeviceMLAnalytics] = None
        self.agent_coordinator: Optional[DSMILAgentCoordinator] = None
        self.db_pool: Optional[asyncpg.Pool] = None
        
        # State
        self.running = False
        self.mode = OrchestrationMode.INTEGRATED
        self.start_time = datetime.now(timezone.utc)
        
        # Performance metrics
        self.total_devices_processed = 0
        self.total_anomalies_detected = 0
        self.total_tasks_coordinated = 0
        
        # Setup logging
        log_config = self.config.get("logging", {})
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                return self._default_config()
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Provide default configuration"""
        return {
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
                "thermal_critical": 85
            },
            "ml": {
                "embedding_dimensions": 512,
                "similarity_threshold": 0.8
            },
            "agents": {
                "max_concurrent": 10,
                "default_timeout": 30.0
            }
        }
    
    async def initialize(self, mode: str = OrchestrationMode.INTEGRATED) -> bool:
        """Initialize the orchestrator system"""
        try:
            self.mode = mode
            self.logger.info(f"Initializing DSMIL ML Orchestrator in {mode} mode...")
            
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
            
            self.logger.info("Database connection pool initialized")
            
            # Initialize components based on mode
            if mode in [OrchestrationMode.MONITORING, OrchestrationMode.INTEGRATED]:
                self.learning_integrator = DSMILLearningIntegrator(self.config_path)
                if not await self.learning_integrator.initialize():
                    self.logger.error("Failed to initialize learning integrator")
                    return False
                self.logger.info("Learning integrator initialized")
            
            if mode in [OrchestrationMode.ANALYSIS, OrchestrationMode.INTEGRATED]:
                self.ml_analytics = DeviceMLAnalytics(self.db_pool, self.config)
                self.logger.info("ML analytics initialized")
            
            if mode in [OrchestrationMode.COORDINATION, OrchestrationMode.INTEGRATED]:
                self.agent_coordinator = DSMILAgentCoordinator(self.db_pool, self.config)
                if not await self.agent_coordinator.initialize():
                    self.logger.error("Failed to initialize agent coordinator")
                    return False
                self.logger.info("Agent coordinator initialized")
            
            # Start background tasks
            asyncio.create_task(self._monitoring_loop())
            if mode in [OrchestrationMode.ANALYSIS, OrchestrationMode.INTEGRATED]:
                asyncio.create_task(self._analysis_loop())
            if mode in [OrchestrationMode.COORDINATION, OrchestrationMode.INTEGRATED]:
                asyncio.create_task(self._coordination_loop())
            
            self.running = True
            self.logger.info("DSMIL ML Orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            return False
    
    async def _monitoring_loop(self) -> None:
        """Background loop for device monitoring and data collection"""
        interval = self.config.get("dsmil", {}).get("monitor_interval", 1.0)
        
        while self.running:
            try:
                if self.learning_integrator:
                    # Simulate device monitoring (replace with actual DSMIL device polling)
                    device_data = await self._collect_device_data()
                    
                    for device_id, data in device_data.items():
                        success = await self.learning_integrator.record_device_pattern(data)
                        if success:
                            self.total_devices_processed += 1
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval * 2)
    
    async def _analysis_loop(self) -> None:
        """Background loop for ML analysis and anomaly detection"""
        analysis_interval = 30.0  # Run analysis every 30 seconds
        
        while self.running:
            try:
                if self.ml_analytics:
                    # Run device pattern analysis
                    analysis_results = await self.ml_analytics.analyze_device_patterns(
                        hours_back=1
                    )
                    
                    if analysis_results:
                        self.logger.info(f"Analyzed patterns for {len(analysis_results)} devices")
                    
                    # Detect anomalies
                    anomalies = await self.ml_analytics.detect_anomalies(
                        severity_threshold=0.7
                    )
                    
                    if anomalies:
                        self.total_anomalies_detected += len(anomalies)
                        self.logger.warning(f"Detected {len(anomalies)} anomalies")
                        
                        # Create tasks for critical anomalies
                        if self.agent_coordinator:
                            await self._handle_anomaly_alerts(anomalies)
                
                await asyncio.sleep(analysis_interval)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(analysis_interval)
    
    async def _coordination_loop(self) -> None:
        """Background loop for coordinating proactive maintenance tasks"""
        coordination_interval = 60.0  # Run coordination every minute
        
        while self.running:
            try:
                if self.agent_coordinator:
                    # Create proactive maintenance tasks based on system state
                    maintenance_tasks = await self._generate_maintenance_tasks()
                    
                    for task in maintenance_tasks:
                        success = await self.agent_coordinator.submit_task(task)
                        if success:
                            self.total_tasks_coordinated += 1
                
                await asyncio.sleep(coordination_interval)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                await asyncio.sleep(coordination_interval)
    
    async def _collect_device_data(self) -> Dict[int, Dict[str, Any]]:
        """Collect current device data (mock implementation - replace with actual DSMIL interface)"""
        # This would normally interface with DSMIL monitoring system
        # For now, generate realistic mock data
        import random
        
        devices = {}
        for device_id in range(1, 11):  # Mock 10 devices
            temp_base = 70 + (device_id % 3) * 5  # Different base temps
            cpu_base = 50 + (device_id % 4) * 10   # Different base CPU usage
            
            devices[device_id] = {
                "device_id": device_id,
                "name": f"DSMIL_Device_{device_id:02d}",
                "temperature": temp_base + random.uniform(-5, 15),
                "cpu_usage": max(0, min(100, cpu_base + random.uniform(-20, 30))),
                "memory_usage": random.uniform(30, 85),
                "disk_usage": random.uniform(10, 70),
                "network_io": random.uniform(0, 100),
                "error_count": random.randint(0, 3) if random.random() > 0.8 else 0,
                "response_time": random.uniform(50, 300),
                "power_consumption": random.uniform(15, 85),
                "status": random.choice(["normal", "normal", "normal", "warning"])
            }
        
        return devices
    
    async def _handle_anomaly_alerts(self, anomalies: List[AnomalyAlert]) -> None:
        """Handle anomaly alerts by creating appropriate tasks"""
        for anomaly in anomalies:
            if anomaly.severity in ["critical", "high"]:
                # Create high-priority task for serious anomalies
                task = TaskRequest(
                    task_id=f"anomaly_response_{anomaly.device_id}_{int(time.time())}",
                    description=f"Investigate and resolve {anomaly.anomaly_type} on {anomaly.device_name}",
                    priority=TaskPriority.HIGH if anomaly.severity == "high" else TaskPriority.EMERGENCY,
                    device_context={
                        "device_id": anomaly.device_id,
                        "device_name": anomaly.device_name,
                        "anomaly_type": anomaly.anomaly_type,
                        "confidence": anomaly.confidence
                    },
                    required_capabilities=["hardware_diagnostics", "performance_analysis"]
                )
                
                if self.agent_coordinator:
                    await self.agent_coordinator.submit_task(task)
                    self.logger.info(f"Created {anomaly.severity} priority task for {anomaly.device_name}")
    
    async def _generate_maintenance_tasks(self) -> List[TaskRequest]:
        """Generate proactive maintenance tasks based on system analysis"""
        tasks = []
        
        try:
            if not self.ml_analytics:
                return tasks
            
            # Get system analytics summary
            summary = await self.ml_analytics.get_analytics_summary()
            
            # Check if we should run performance optimization
            if summary.get("performance", {}).get("total_analyses", 0) > 100:
                throughput = summary.get("performance", {}).get("throughput_per_hour", 0)
                
                if throughput < 3600:  # Less than optimal throughput
                    tasks.append(TaskRequest(
                        task_id=f"performance_optimization_{int(time.time())}",
                        description="Analyze and optimize system performance based on ML analytics",
                        priority=TaskPriority.MEDIUM,
                        required_capabilities=["performance_optimization", "system_analysis"]
                    ))
            
            # Check for security review needs
            if self.total_anomalies_detected > 50:  # Many anomalies detected
                tasks.append(TaskRequest(
                    task_id=f"security_review_{int(time.time())}",
                    description="Comprehensive security review due to high anomaly detection rate",
                    priority=TaskPriority.HIGH,
                    required_capabilities=["security_analysis", "threat_assessment"]
                ))
            
            # Regular system health check
            uptime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
            if uptime_hours > 24 and uptime_hours % 24 < 1:  # Every 24 hours
                tasks.append(TaskRequest(
                    task_id=f"daily_health_check_{int(time.time())}",
                    description="Daily comprehensive system health check and maintenance",
                    priority=TaskPriority.LOW,
                    required_capabilities=["system_monitoring", "maintenance"]
                ))
            
            return tasks
            
        except Exception as e:
            self.logger.error(f"Failed to generate maintenance tasks: {e}")
            return []
    
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health status"""
        try:
            # Learning system status
            learning_status = "offline"
            if self.learning_integrator:
                learning_health = await self.learning_integrator.get_system_health_summary()
                learning_status = learning_health.get("status", "unknown")
            
            # ML analytics status
            ml_status = "offline"
            if self.ml_analytics:
                analytics_summary = await self.ml_analytics.get_analytics_summary()
                ml_status = analytics_summary.get("status", "unknown")
            
            # Agent coordinator status
            coordinator_status = "offline"
            active_agents = 0
            recent_success_rate = 0.0
            if self.agent_coordinator:
                coord_status = await self.agent_coordinator.get_coordination_status()
                coordinator_status = coord_status.get("status", "unknown")
                active_agents = coord_status.get("current_state", {}).get("idle_agents", 0)
                recent_success_rate = coord_status.get("performance", {}).get("recent_success_rate", 0.0)
            
            # Database connection test
            db_connected = False
            if self.db_pool:
                try:
                    async with self.db_pool.acquire() as conn:
                        await conn.fetchval("SELECT 1")
                    db_connected = True
                except:
                    pass
            
            return SystemHealth(
                learning_system_status=learning_status,
                ml_analytics_status=ml_status,
                agent_coordinator_status=coordinator_status,
                database_connection=db_connected,
                total_devices_monitored=self.total_devices_processed,
                active_agents=active_agents,
                recent_success_rate=recent_success_rate,
                avg_response_time=0.0,  # Would calculate from metrics
                anomalies_detected=self.total_anomalies_detected,
                last_update=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return SystemHealth(
                learning_system_status="error",
                ml_analytics_status="error",
                agent_coordinator_status="error",
                database_connection=False,
                total_devices_monitored=0,
                active_agents=0,
                recent_success_rate=0.0,
                avg_response_time=0.0,
                anomalies_detected=0,
                last_update=datetime.now(timezone.utc)
            )
    
    async def submit_manual_task(self, description: str, priority: str = "medium",
                               device_context: Optional[Dict] = None) -> bool:
        """Submit a manual task for agent coordination"""
        try:
            if not self.agent_coordinator:
                self.logger.error("Agent coordinator not initialized")
                return False
            
            priority_map = {
                "emergency": TaskPriority.EMERGENCY,
                "high": TaskPriority.HIGH,
                "medium": TaskPriority.MEDIUM,
                "low": TaskPriority.LOW,
                "background": TaskPriority.BACKGROUND
            }
            
            task = TaskRequest(
                task_id=f"manual_task_{int(time.time())}",
                description=description,
                priority=priority_map.get(priority.lower(), TaskPriority.MEDIUM),
                device_context=device_context
            )
            
            return await self.agent_coordinator.submit_task(task)
            
        except Exception as e:
            self.logger.error(f"Failed to submit manual task: {e}")
            return False
    
    async def get_agent_recommendations(self, task_description: str,
                                      device_context: Optional[Dict] = None) -> List[Dict]:
        """Get agent recommendations for a task"""
        if not self.agent_coordinator:
            return []
        
        return await self.agent_coordinator.get_agent_recommendations(
            task_description, device_context
        )
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the orchestrator"""
        self.logger.info("Shutting down DSMIL ML Orchestrator...")
        
        self.running = False
        
        # Shutdown components
        if self.learning_integrator:
            await self.learning_integrator.shutdown()
        
        if self.db_pool:
            await self.db_pool.close()
        
        self.logger.info("Shutdown complete")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False

# CLI Interface
async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="DSMIL ML Orchestrator")
    parser.add_argument("--mode", choices=["monitoring", "analysis", "coordination", "integrated"],
                       default="integrated", help="Orchestration mode")
    parser.add_argument("--config", type=Path, help="Configuration file path")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    parser.add_argument("--task", type=str, help="Submit a manual task")
    parser.add_argument("--priority", choices=["emergency", "high", "medium", "low", "background"],
                       default="medium", help="Task priority")
    parser.add_argument("--recommend", type=str, help="Get agent recommendations for a task")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DSMILMLOrchestrator(args.config)
    
    try:
        if not await orchestrator.initialize(args.mode):
            print("Failed to initialize orchestrator")
            return 1
        
        if args.status:
            # Show status and exit
            health = await orchestrator.get_system_health()
            print("\n=== DSMIL ML Orchestrator Status ===")
            print(f"Learning System: {health.learning_system_status}")
            print(f"ML Analytics: {health.ml_analytics_status}")
            print(f"Agent Coordinator: {health.agent_coordinator_status}")
            print(f"Database Connected: {health.database_connection}")
            print(f"Devices Monitored: {health.total_devices_monitored}")
            print(f"Active Agents: {health.active_agents}")
            print(f"Success Rate: {health.recent_success_rate:.1f}%")
            print(f"Anomalies Detected: {health.anomalies_detected}")
            print(f"Last Update: {health.last_update.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            return 0
        
        if args.task:
            # Submit manual task and exit
            success = await orchestrator.submit_manual_task(args.task, args.priority)
            if success:
                print(f"Task submitted successfully with {args.priority} priority")
                return 0
            else:
                print("Failed to submit task")
                return 1
        
        if args.recommend:
            # Get recommendations and exit
            recommendations = await orchestrator.get_agent_recommendations(args.recommend)
            print(f"\n=== Agent Recommendations for: {args.recommend} ===")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"{i}. {rec['agent_name']} ({rec['category']})")
                print(f"   Confidence: {rec['confidence']:.3f}")
                print(f"   Duration: {rec['estimated_duration']:.1f}s")
                print(f"   Specializations: {', '.join(rec['specializations'][:3])}")
                print()
            return 0
        
        # Run orchestrator
        print(f"Starting DSMIL ML Orchestrator in {args.mode} mode...")
        print("Press Ctrl+C to stop")
        
        while orchestrator.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        await orchestrator.shutdown()
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))