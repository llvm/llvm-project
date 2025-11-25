#!/usr/bin/env python3
"""
NPU/GNA Hardware Performance and Health Monitoring System
Comprehensive monitoring of hardware acceleration components

Author: Hardware Health Monitoring Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import psutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
import uuid
from collections import deque, defaultdict
import ctypes
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HardwareComponent(Enum):
    """Hardware acceleration components"""
    NPU = "npu"
    GNA = "gna"
    AVX2 = "avx2"
    AVX512 = "avx512"
    AES_NI = "aes_ni"
    CPU_CRYPTO = "cpu_crypto"
    INTEL_QAT = "intel_qat"
    GPU_COMPUTE = "gpu_compute"

class HealthStatus(Enum):
    """Hardware health status levels"""
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"
    OFFLINE = "offline"

class PerformanceMetric(Enum):
    """Performance metrics"""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    UTILIZATION = "utilization"
    POWER_CONSUMPTION = "power_consumption"
    TEMPERATURE = "temperature"
    ERROR_RATE = "error_rate"
    BANDWIDTH = "bandwidth"
    EFFICIENCY = "efficiency"

@dataclass
class HardwareDevice:
    """Hardware device information"""
    device_id: str
    component_type: HardwareComponent
    device_path: str
    driver_name: str
    driver_version: str
    firmware_version: str
    hardware_revision: str
    vendor_id: str
    device_id_hex: str
    capabilities: List[str]
    supported_operations: List[str]
    max_performance: Dict[str, float]
    power_characteristics: Dict[str, float]

@dataclass
class PerformanceMeasurement:
    """Performance measurement record"""
    measurement_id: str
    timestamp: float
    device_id: str
    metric: PerformanceMetric
    value: float
    unit: str
    test_workload: str
    baseline_value: Optional[float]
    deviation_percent: Optional[float]
    status: HealthStatus

@dataclass
class HealthAssessment:
    """Hardware health assessment"""
    assessment_id: str
    timestamp: float
    device_id: str
    overall_status: HealthStatus
    component_scores: Dict[str, float]
    performance_grade: str
    reliability_score: float
    degradation_indicators: List[str]
    failure_predictions: List[Dict[str, Any]]
    maintenance_recommendations: List[str]
    estimated_lifetime: Optional[float]

@dataclass
class FallbackEvent:
    """Hardware fallback event"""
    event_id: str
    timestamp: float
    from_component: HardwareComponent
    to_component: HardwareComponent
    trigger_reason: str
    performance_impact: float
    duration_seconds: float
    recovery_status: str
    automatic_fallback: bool

class HardwareHealthMonitor:
    """
    Comprehensive hardware health and performance monitoring system
    Monitors NPU, GNA, and other acceleration hardware
    """

    def __init__(self, config_path: str = "/etc/military-tpm/hardware_monitor.json"):
        """Initialize hardware health monitor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False

        # Database for persistent storage
        self.db_path = self.config.get("database_path", "/var/lib/military-tpm/hardware_health.db")
        self._init_database()

        # Hardware discovery
        self.discovered_devices = {}
        self.device_capabilities = {}
        self.performance_baselines = {}

        # Monitoring state
        self.performance_history = defaultdict(lambda: deque(maxlen=3600))  # 1 hour per metric
        self.health_assessments = deque(maxlen=1000)
        self.fallback_events = deque(maxlen=500)
        self.active_tests = {}

        # Predictive analytics
        self.degradation_models = {}
        self.failure_predictors = {}
        self.maintenance_scheduler = {}

        # Monitoring threads
        self.monitoring_threads = []

        # Discover and initialize hardware
        self._discover_hardware()
        self._initialize_baselines()

        logger.info("Hardware Health Monitor initialized")

    def start(self):
        """Start hardware health monitoring"""
        logger.info("Starting hardware health monitoring...")
        self.running = True

        # Start monitoring threads
        self._start_device_monitoring()
        self._start_performance_testing()
        self._start_health_assessment()
        self._start_predictive_analysis()
        self._start_fallback_detection()

        logger.info("Hardware health monitoring started")

    def stop(self):
        """Stop hardware health monitoring"""
        logger.info("Stopping hardware health monitoring...")
        self.running = False

        # Stop all monitoring threads
        for thread in self.monitoring_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        logger.info("Hardware health monitoring stopped")

    def get_hardware_status(self) -> Dict[str, Any]:
        """Get comprehensive hardware status"""
        status = {
            "timestamp": time.time(),
            "discovered_devices": len(self.discovered_devices),
            "device_status": {},
            "overall_health": self._calculate_overall_health(),
            "performance_summary": self._get_performance_summary(),
            "active_fallbacks": self._get_active_fallbacks(),
            "recent_degradations": self._get_recent_degradations(),
            "maintenance_alerts": self._get_maintenance_alerts()
        }

        # Device-specific status
        for device_id, device in self.discovered_devices.items():
            latest_assessment = self._get_latest_assessment(device_id)
            status["device_status"][device_id] = {
                "component_type": device.component_type.value,
                "health_status": latest_assessment.overall_status.value if latest_assessment else "unknown",
                "performance_grade": latest_assessment.performance_grade if latest_assessment else "unknown",
                "reliability_score": latest_assessment.reliability_score if latest_assessment else 0.0,
                "device_path": device.device_path,
                "driver_version": device.driver_version
            }

        return status

    def run_performance_benchmark(self, device_id: str, workload_type: str = "standard") -> str:
        """Run comprehensive performance benchmark on device"""
        benchmark_id = str(uuid.uuid4())
        timestamp = time.time()

        if device_id not in self.discovered_devices:
            raise ValueError(f"Device not found: {device_id}")

        device = self.discovered_devices[device_id]

        try:
            # Log benchmark start
            logger.info(f"Starting performance benchmark {benchmark_id} on {device_id}")

            # Run component-specific benchmarks
            results = self._run_component_benchmark(device, workload_type)

            # Store benchmark results
            self._store_benchmark_results(benchmark_id, device_id, workload_type, results)

            # Update performance baselines if needed
            self._update_performance_baselines(device_id, results)

            # Assess performance degradation
            degradation = self._assess_performance_degradation(device_id, results)

            logger.info(f"Performance benchmark {benchmark_id} completed for {device_id}")
            return benchmark_id

        except Exception as e:
            logger.error(f"Error running benchmark {benchmark_id} on {device_id}: {e}")
            raise

    def predict_hardware_failure(self, device_id: str, prediction_horizon_days: int = 30) -> Dict[str, Any]:
        """Predict potential hardware failures"""
        if device_id not in self.discovered_devices:
            raise ValueError(f"Device not found: {device_id}")

        # Get historical performance data
        historical_data = self._get_historical_performance_data(device_id, days=90)

        # Run predictive models
        predictions = {
            "device_id": device_id,
            "prediction_horizon_days": prediction_horizon_days,
            "failure_probability": 0.0,
            "predicted_failure_modes": [],
            "confidence_score": 0.0,
            "recommended_actions": [],
            "estimated_remaining_life": None
        }

        try:
            # Temperature-based predictions
            temp_prediction = self._predict_temperature_failure(historical_data, prediction_horizon_days)
            predictions["failure_probability"] = max(predictions["failure_probability"], temp_prediction["probability"])

            # Performance degradation predictions
            perf_prediction = self._predict_performance_degradation(historical_data, prediction_horizon_days)
            predictions["failure_probability"] = max(predictions["failure_probability"], perf_prediction["probability"])

            # Error rate trend predictions
            error_prediction = self._predict_error_rate_increase(historical_data, prediction_horizon_days)
            predictions["failure_probability"] = max(predictions["failure_probability"], error_prediction["probability"])

            # Combine predictions
            predictions["predicted_failure_modes"] = (
                temp_prediction.get("failure_modes", []) +
                perf_prediction.get("failure_modes", []) +
                error_prediction.get("failure_modes", [])
            )

            # Calculate confidence score
            predictions["confidence_score"] = self._calculate_prediction_confidence(
                historical_data, predictions["failure_probability"]
            )

            # Generate recommendations
            predictions["recommended_actions"] = self._generate_maintenance_recommendations(
                device_id, predictions["failure_probability"], predictions["predicted_failure_modes"]
            )

            # Estimate remaining life
            if predictions["failure_probability"] > 0.1:
                predictions["estimated_remaining_life"] = self._estimate_remaining_life(
                    device_id, historical_data, predictions["failure_probability"]
                )

        except Exception as e:
            logger.error(f"Error predicting hardware failure for {device_id}: {e}")

        return predictions

    def _discover_hardware(self):
        """Discover available hardware acceleration devices"""
        logger.info("Discovering hardware acceleration devices...")

        # Discover NPU devices
        self._discover_npu_devices()

        # Discover GNA devices
        self._discover_gna_devices()

        # Discover CPU acceleration features
        self._discover_cpu_acceleration()

        # Discover Intel QAT devices
        self._discover_qat_devices()

        # Discover GPU compute devices
        self._discover_gpu_devices()

        logger.info(f"Discovered {len(self.discovered_devices)} hardware acceleration devices")

    def _discover_npu_devices(self):
        """Discover NPU (Neural Processing Unit) devices"""
        npu_paths = [
            '/dev/intel_npu',
            '/dev/npu0',
            '/dev/accel/accel0',
            '/sys/class/accel/accel*'
        ]

        for pattern in npu_paths:
            for device_path in glob.glob(pattern):
                if os.path.exists(device_path):
                    try:
                        device_info = self._get_npu_device_info(device_path)
                        if device_info:
                            device_id = f"npu_{len([d for d in self.discovered_devices.values() if d.component_type == HardwareComponent.NPU])}"
                            device = HardwareDevice(
                                device_id=device_id,
                                component_type=HardwareComponent.NPU,
                                device_path=device_path,
                                driver_name=device_info.get("driver", "unknown"),
                                driver_version=device_info.get("driver_version", "unknown"),
                                firmware_version=device_info.get("firmware_version", "unknown"),
                                hardware_revision=device_info.get("hardware_revision", "unknown"),
                                vendor_id=device_info.get("vendor_id", "unknown"),
                                device_id_hex=device_info.get("device_id", "unknown"),
                                capabilities=device_info.get("capabilities", []),
                                supported_operations=device_info.get("supported_operations", []),
                                max_performance=device_info.get("max_performance", {}),
                                power_characteristics=device_info.get("power_characteristics", {})
                            )
                            self.discovered_devices[device_id] = device
                            logger.info(f"Discovered NPU device: {device_id} at {device_path}")
                    except Exception as e:
                        logger.warning(f"Error discovering NPU device at {device_path}: {e}")

    def _discover_gna_devices(self):
        """Discover GNA (Gaussian & Neural Accelerator) devices"""
        gna_paths = [
            '/dev/gna0',
            '/dev/gna*',
            '/sys/class/gna/gna*'
        ]

        for pattern in gna_paths:
            for device_path in glob.glob(pattern):
                if os.path.exists(device_path):
                    try:
                        device_info = self._get_gna_device_info(device_path)
                        if device_info:
                            device_id = f"gna_{len([d for d in self.discovered_devices.values() if d.component_type == HardwareComponent.GNA])}"
                            device = HardwareDevice(
                                device_id=device_id,
                                component_type=HardwareComponent.GNA,
                                device_path=device_path,
                                driver_name=device_info.get("driver", "unknown"),
                                driver_version=device_info.get("driver_version", "unknown"),
                                firmware_version=device_info.get("firmware_version", "unknown"),
                                hardware_revision=device_info.get("hardware_revision", "unknown"),
                                vendor_id=device_info.get("vendor_id", "unknown"),
                                device_id_hex=device_info.get("device_id", "unknown"),
                                capabilities=device_info.get("capabilities", []),
                                supported_operations=device_info.get("supported_operations", []),
                                max_performance=device_info.get("max_performance", {}),
                                power_characteristics=device_info.get("power_characteristics", {})
                            )
                            self.discovered_devices[device_id] = device
                            logger.info(f"Discovered GNA device: {device_id} at {device_path}")
                    except Exception as e:
                        logger.warning(f"Error discovering GNA device at {device_path}: {e}")

    def _discover_cpu_acceleration(self):
        """Discover CPU acceleration features"""
        try:
            # Check CPU features
            cpu_features = self._get_cpu_features()

            # Create devices for each acceleration feature
            if "avx2" in cpu_features:
                device_id = "cpu_avx2"
                device = HardwareDevice(
                    device_id=device_id,
                    component_type=HardwareComponent.AVX2,
                    device_path="/proc/cpuinfo",
                    driver_name="cpu",
                    driver_version="kernel",
                    firmware_version="microcode",
                    hardware_revision=cpu_features.get("model", "unknown"),
                    vendor_id=cpu_features.get("vendor_id", "unknown"),
                    device_id_hex=cpu_features.get("model_id", "unknown"),
                    capabilities=["avx2", "vector_operations"],
                    supported_operations=["crypto", "math", "data_processing"],
                    max_performance={"throughput_gops": 100.0, "power_watts": 15.0},
                    power_characteristics={"tdp_watts": 65.0, "idle_watts": 5.0}
                )
                self.discovered_devices[device_id] = device

            if "avx512" in cpu_features:
                device_id = "cpu_avx512"
                device = HardwareDevice(
                    device_id=device_id,
                    component_type=HardwareComponent.AVX512,
                    device_path="/proc/cpuinfo",
                    driver_name="cpu",
                    driver_version="kernel",
                    firmware_version="microcode",
                    hardware_revision=cpu_features.get("model", "unknown"),
                    vendor_id=cpu_features.get("vendor_id", "unknown"),
                    device_id_hex=cpu_features.get("model_id", "unknown"),
                    capabilities=["avx512", "vector_operations", "high_throughput"],
                    supported_operations=["crypto", "math", "data_processing", "ai_inference"],
                    max_performance={"throughput_gops": 200.0, "power_watts": 25.0},
                    power_characteristics={"tdp_watts": 95.0, "idle_watts": 5.0}
                )
                self.discovered_devices[device_id] = device

            if "aes" in cpu_features:
                device_id = "cpu_aes_ni"
                device = HardwareDevice(
                    device_id=device_id,
                    component_type=HardwareComponent.AES_NI,
                    device_path="/proc/cpuinfo",
                    driver_name="cpu",
                    driver_version="kernel",
                    firmware_version="microcode",
                    hardware_revision=cpu_features.get("model", "unknown"),
                    vendor_id=cpu_features.get("vendor_id", "unknown"),
                    device_id_hex=cpu_features.get("model_id", "unknown"),
                    capabilities=["aes_ni", "hardware_crypto"],
                    supported_operations=["aes_encrypt", "aes_decrypt", "crypto_hash"],
                    max_performance={"throughput_mbps": 5000.0, "latency_us": 1.0},
                    power_characteristics={"crypto_power_watts": 2.0, "idle_watts": 0.1}
                )
                self.discovered_devices[device_id] = device

        except Exception as e:
            logger.error(f"Error discovering CPU acceleration features: {e}")

    def _get_npu_device_info(self, device_path: str) -> Optional[Dict[str, Any]]:
        """Get NPU device information"""
        try:
            # Check if device is accessible
            if not os.access(device_path, os.R_OK):
                return None

            device_info = {
                "driver": "intel_npu",
                "driver_version": self._get_driver_version("intel_npu"),
                "firmware_version": "unknown",
                "hardware_revision": "unknown",
                "vendor_id": "8086",  # Intel
                "device_id": "unknown",
                "capabilities": ["neural_inference", "low_power", "real_time"],
                "supported_operations": ["convolution", "fully_connected", "pooling", "activation"],
                "max_performance": {
                    "tops": 2.0,  # Typical NPU performance
                    "power_watts": 1.0,
                    "latency_ms": 1.0
                },
                "power_characteristics": {
                    "max_power_watts": 2.0,
                    "idle_power_watts": 0.1,
                    "efficiency_tops_per_watt": 2.0
                }
            }

            # Try to get more specific information from sysfs
            sysfs_path = device_path.replace('/dev', '/sys/class').replace('/accel', '/accel')
            if os.path.exists(sysfs_path):
                # Read device properties
                try:
                    with open(f"{sysfs_path}/device/vendor", 'r') as f:
                        device_info["vendor_id"] = f.read().strip()
                except:
                    pass

                try:
                    with open(f"{sysfs_path}/device/device", 'r') as f:
                        device_info["device_id"] = f.read().strip()
                except:
                    pass

            return device_info

        except Exception as e:
            logger.error(f"Error getting NPU device info for {device_path}: {e}")
            return None

    def _get_gna_device_info(self, device_path: str) -> Optional[Dict[str, Any]]:
        """Get GNA device information"""
        try:
            # Check if device is accessible
            if not os.access(device_path, os.R_OK):
                return None

            device_info = {
                "driver": "gna",
                "driver_version": self._get_driver_version("gna"),
                "firmware_version": "unknown",
                "hardware_revision": "unknown",
                "vendor_id": "8086",  # Intel
                "device_id": "unknown",
                "capabilities": ["gaussian_mixture", "neural_network", "low_power"],
                "supported_operations": ["dnn", "gmm", "feature_extraction"],
                "max_performance": {
                    "gops": 1.0,  # Typical GNA performance
                    "power_watts": 0.5,
                    "latency_ms": 2.0
                },
                "power_characteristics": {
                    "max_power_watts": 1.0,
                    "idle_power_watts": 0.05,
                    "efficiency_gops_per_watt": 2.0
                }
            }

            return device_info

        except Exception as e:
            logger.error(f"Error getting GNA device info for {device_path}: {e}")
            return None

    def _get_cpu_features(self) -> Dict[str, Any]:
        """Get CPU features and capabilities"""
        features = {}

        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()

            # Extract CPU features
            for line in cpuinfo.split('\n'):
                if line.startswith('flags') or line.startswith('Features'):
                    flags = line.split(':')[1].strip().split()
                    if 'avx2' in flags:
                        features['avx2'] = True
                    if 'avx512f' in flags:
                        features['avx512'] = True
                    if 'aes' in flags:
                        features['aes'] = True
                elif line.startswith('vendor_id'):
                    features['vendor_id'] = line.split(':')[1].strip()
                elif line.startswith('model name'):
                    features['model'] = line.split(':')[1].strip()
                elif line.startswith('model'):
                    features['model_id'] = line.split(':')[1].strip()

        except Exception as e:
            logger.error(f"Error reading CPU features: {e}")

        return features

    def _get_driver_version(self, driver_name: str) -> str:
        """Get driver version"""
        try:
            result = subprocess.run(['modinfo', driver_name], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('version:'):
                        return line.split(':')[1].strip()
        except:
            pass
        return "unknown"

    # Additional implementation methods would continue here...
    # This includes methods for:
    # - Performance benchmarking
    # - Health assessment algorithms
    # - Predictive failure analysis
    # - Temperature monitoring
    # - Power consumption tracking
    # - Fallback detection
    # - etc.


def main():
    """Main hardware health monitor entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Hardware Health Monitor")
    parser.add_argument("--config", default="/etc/military-tpm/hardware_monitor.json",
                       help="Configuration file path")
    parser.add_argument("--status", action="store_true",
                       help="Show hardware status and exit")
    parser.add_argument("--benchmark", metavar="DEVICE_ID",
                       help="Run performance benchmark on device")
    parser.add_argument("--predict-failure", metavar="DEVICE_ID",
                       help="Predict hardware failure for device")

    args = parser.parse_args()

    # Create hardware monitor
    monitor = HardwareHealthMonitor(args.config)

    if args.status:
        # Show status and exit
        monitor.start()
        time.sleep(2)  # Allow device discovery
        status = monitor.get_hardware_status()
        print(json.dumps(status, indent=2))
        monitor.stop()
        return

    if args.benchmark:
        # Run benchmark and exit
        monitor.start()
        time.sleep(2)  # Allow device discovery
        benchmark_id = monitor.run_performance_benchmark(args.benchmark)
        print(f"Benchmark completed: {benchmark_id}")
        monitor.stop()
        return

    if args.predict_failure:
        # Predict failure and exit
        monitor.start()
        time.sleep(2)  # Allow device discovery
        prediction = monitor.predict_hardware_failure(args.predict_failure)
        print(json.dumps(prediction, indent=2))
        monitor.stop()
        return

    # Run hardware monitor
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        monitor.start()

        # Keep running until signal received
        while monitor.running:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()