#!/usr/bin/env python3
"""
TPM2 Acceleration Health Monitor
Monitors hardware acceleration health and manages fallback mechanisms

Author: TPM2 Health Monitor Agent
Date: 2025-09-23
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import json
import logging
import threading
import signal
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

class AccelerationType(Enum):
    """Acceleration types"""
    NPU = "npu"
    GNA = "gna"
    CPU_OPTIMIZED = "cpu_optimized"
    CPU_BASIC = "cpu_basic"

@dataclass
class HealthMetrics:
    """Health metrics for acceleration hardware"""
    acceleration_type: AccelerationType
    status: HealthStatus
    response_time_ms: float
    throughput_ops_sec: float
    error_rate_percent: float
    temperature_celsius: Optional[float]
    utilization_percent: float
    last_checked: float

@dataclass
class FallbackEvent:
    """Fallback activation event"""
    timestamp: float
    from_acceleration: AccelerationType
    to_acceleration: AccelerationType
    reason: str
    recovery_time_ms: float

class AccelerationHealthMonitor:
    """
    Monitors acceleration hardware health and manages automatic fallback
    """

    def __init__(self, config_path: str = "/etc/military-tpm/fallback.json"):
        """Initialize health monitor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.current_acceleration = AccelerationType.CPU_BASIC
        self.health_metrics = {}
        self.fallback_events = []
        self.failure_counts = {}
        self.monitoring_thread = None

        # Initialize failure counters
        for accel_type in AccelerationType:
            self.failure_counts[accel_type] = 0

        logger.info("Acceleration Health Monitor initialized")

    def start(self):
        """Start health monitoring"""
        logger.info("Starting acceleration health monitoring...")
        self.running = True

        # Determine initial acceleration type
        self._determine_initial_acceleration()

        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info(f"Health monitoring started with {self.current_acceleration.value} acceleration")

    def stop(self):
        """Stop health monitoring"""
        logger.info("Stopping acceleration health monitoring...")
        self.running = False

        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)

        logger.info("Health monitoring stopped")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        return {
            "current_acceleration": self.current_acceleration.value,
            "health_metrics": {
                accel.value: {
                    "status": metrics.status.value,
                    "response_time_ms": metrics.response_time_ms,
                    "throughput_ops_sec": metrics.throughput_ops_sec,
                    "error_rate_percent": metrics.error_rate_percent,
                    "utilization_percent": metrics.utilization_percent,
                    "last_checked": metrics.last_checked
                }
                for accel, metrics in self.health_metrics.items()
            },
            "failure_counts": {accel.value: count for accel, count in self.failure_counts.items()},
            "recent_fallback_events": [
                {
                    "timestamp": event.timestamp,
                    "from": event.from_acceleration.value,
                    "to": event.to_acceleration.value,
                    "reason": event.reason,
                    "recovery_time_ms": event.recovery_time_ms
                }
                for event in self.fallback_events[-10:]  # Last 10 events
            ]
        }

    def force_fallback(self, target_acceleration: AccelerationType, reason: str = "Manual override"):
        """Force fallback to specific acceleration type"""
        logger.info(f"Forcing fallback to {target_acceleration.value}: {reason}")
        self._activate_fallback(target_acceleration, reason)

    def _load_config(self) -> Dict[str, Any]:
        """Load fallback configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "enabled": True,
            "automatic_detection": True,
            "health_check_interval_ms": 5000,
            "failure_threshold": 3,
            "recovery_timeout_ms": 30000,
            "fallback_chain": [
                {"type": "npu", "priority": 1, "enabled": True},
                {"type": "gna", "priority": 2, "enabled": True},
                {"type": "cpu_optimized", "priority": 3, "enabled": True},
                {"type": "cpu_basic", "priority": 4, "enabled": True}
            ]
        }

    def _determine_initial_acceleration(self):
        """Determine initial acceleration type based on availability"""
        fallback_chain = sorted(
            self.config.get("fallback_chain", []),
            key=lambda x: x.get("priority", 999)
        )

        for fallback_config in fallback_chain:
            if not fallback_config.get("enabled", False):
                continue

            accel_type_str = fallback_config.get("type", "")
            try:
                accel_type = AccelerationType(accel_type_str)
                if self._test_acceleration_availability(accel_type):
                    self.current_acceleration = accel_type
                    logger.info(f"Initial acceleration: {accel_type.value}")
                    return
            except ValueError:
                logger.warning(f"Unknown acceleration type in config: {accel_type_str}")

        # Fallback to CPU basic if nothing else works
        self.current_acceleration = AccelerationType.CPU_BASIC
        logger.warning("Falling back to CPU basic acceleration")

    def _test_acceleration_availability(self, acceleration: AccelerationType) -> bool:
        """Test if acceleration type is available"""
        try:
            if acceleration == AccelerationType.NPU:
                return self._test_npu_availability()
            elif acceleration == AccelerationType.GNA:
                return self._test_gna_availability()
            elif acceleration == AccelerationType.CPU_OPTIMIZED:
                return self._test_cpu_optimized_availability()
            else:  # CPU_BASIC
                return True  # Always available
        except Exception as e:
            logger.warning(f"Error testing {acceleration.value} availability: {e}")
            return False

    def _test_npu_availability(self) -> bool:
        """Test NPU availability"""
        # Check for NPU device
        npu_devices = ['/dev/intel_npu', '/dev/npu0', '/dev/accel/accel0']
        for device in npu_devices:
            if os.path.exists(device):
                try:
                    # Test basic access
                    with open(device, 'rb') as f:
                        pass
                    return True
                except PermissionError:
                    logger.warning(f"NPU device {device} permission denied")
                except Exception as e:
                    logger.warning(f"NPU device {device} test failed: {e}")
        return False

    def _test_gna_availability(self) -> bool:
        """Test GNA availability"""
        # Check for GNA device
        if os.path.exists('/dev/gna0'):
            try:
                with open('/dev/gna0', 'rb') as f:
                    pass
                return True
            except Exception as e:
                logger.warning(f"GNA device test failed: {e}")
        return False

    def _test_cpu_optimized_availability(self) -> bool:
        """Test CPU optimized availability (AVX2, AES-NI)"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            return 'avx2' in cpuinfo and 'aes' in cpuinfo
        except Exception:
            return False

    def _monitoring_loop(self):
        """Main monitoring loop"""
        check_interval = self.config.get("health_check_interval_ms", 5000) / 1000.0

        while self.running:
            try:
                # Check health of all available acceleration types
                self._check_all_acceleration_health()

                # Evaluate if fallback is needed
                self._evaluate_fallback_need()

                # Sleep until next check
                time.sleep(check_interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)  # Short sleep on error

    def _check_all_acceleration_health(self):
        """Check health of all acceleration types"""
        for accel_type in AccelerationType:
            if self._test_acceleration_availability(accel_type):
                metrics = self._measure_acceleration_health(accel_type)
                self.health_metrics[accel_type] = metrics

    def _measure_acceleration_health(self, acceleration: AccelerationType) -> HealthMetrics:
        """Measure health metrics for specific acceleration"""
        start_time = time.time()

        try:
            # Perform health check operation
            if acceleration == AccelerationType.NPU:
                response_time, throughput, error_rate = self._npu_health_check()
            elif acceleration == AccelerationType.GNA:
                response_time, throughput, error_rate = self._gna_health_check()
            elif acceleration == AccelerationType.CPU_OPTIMIZED:
                response_time, throughput, error_rate = self._cpu_optimized_health_check()
            else:  # CPU_BASIC
                response_time, throughput, error_rate = self._cpu_basic_health_check()

            # Determine status based on metrics
            status = self._determine_health_status(response_time, throughput, error_rate)

            # Get system metrics
            temperature = self._get_cpu_temperature()
            utilization = psutil.cpu_percent(interval=0.1)

            return HealthMetrics(
                acceleration_type=acceleration,
                status=status,
                response_time_ms=response_time,
                throughput_ops_sec=throughput,
                error_rate_percent=error_rate,
                temperature_celsius=temperature,
                utilization_percent=utilization,
                last_checked=time.time()
            )

        except Exception as e:
            logger.error(f"Health check failed for {acceleration.value}: {e}")
            return HealthMetrics(
                acceleration_type=acceleration,
                status=HealthStatus.FAILED,
                response_time_ms=float('inf'),
                throughput_ops_sec=0.0,
                error_rate_percent=100.0,
                temperature_celsius=None,
                utilization_percent=0.0,
                last_checked=time.time()
            )

    def _npu_health_check(self) -> tuple[float, float, float]:
        """Perform NPU health check"""
        start_time = time.time()

        # Simulate NPU operation
        operations = 100
        errors = 0

        try:
            # Test NPU device access
            for i in range(operations):
                # Simulate NPU operation (in practice, would use actual NPU APIs)
                time.sleep(0.001)  # Simulate operation time

        except Exception:
            errors += 1

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000 / operations
        throughput_ops_sec = operations / (end_time - start_time)
        error_rate_percent = (errors / operations) * 100

        return response_time_ms, throughput_ops_sec, error_rate_percent

    def _gna_health_check(self) -> tuple[float, float, float]:
        """Perform GNA health check"""
        start_time = time.time()

        # Simulate GNA operation
        operations = 50
        errors = 0

        try:
            # Test GNA device access
            for i in range(operations):
                # Simulate GNA operation
                time.sleep(0.002)  # Simulate operation time

        except Exception:
            errors += 1

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000 / operations
        throughput_ops_sec = operations / (end_time - start_time)
        error_rate_percent = (errors / operations) * 100

        return response_time_ms, throughput_ops_sec, error_rate_percent

    def _cpu_optimized_health_check(self) -> tuple[float, float, float]:
        """Perform CPU optimized health check"""
        import hashlib

        start_time = time.time()
        operations = 1000
        errors = 0

        try:
            # Test AVX2-optimized operations
            for i in range(operations):
                hashlib.sha256(f"test_data_{i}".encode()).digest()
        except Exception:
            errors += 1

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000 / operations
        throughput_ops_sec = operations / (end_time - start_time)
        error_rate_percent = (errors / operations) * 100

        return response_time_ms, throughput_ops_sec, error_rate_percent

    def _cpu_basic_health_check(self) -> tuple[float, float, float]:
        """Perform CPU basic health check"""
        import hashlib

        start_time = time.time()
        operations = 100
        errors = 0

        try:
            # Test basic CPU operations
            for i in range(operations):
                hashlib.md5(f"test_data_{i}".encode()).digest()
        except Exception:
            errors += 1

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000 / operations
        throughput_ops_sec = operations / (end_time - start_time)
        error_rate_percent = (errors / operations) * 100

        return response_time_ms, throughput_ops_sec, error_rate_percent

    def _determine_health_status(self, response_time_ms: float,
                                throughput_ops_sec: float,
                                error_rate_percent: float) -> HealthStatus:
        """Determine health status based on metrics"""
        # Configurable thresholds
        critical_response_time_ms = 1000
        warning_response_time_ms = 500
        critical_error_rate_percent = 10
        warning_error_rate_percent = 5

        if error_rate_percent >= critical_error_rate_percent:
            return HealthStatus.CRITICAL
        elif response_time_ms >= critical_response_time_ms:
            return HealthStatus.CRITICAL
        elif error_rate_percent >= warning_error_rate_percent:
            return HealthStatus.WARNING
        elif response_time_ms >= warning_response_time_ms:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature if available"""
        try:
            temperatures = psutil.sensors_temperatures()
            if 'coretemp' in temperatures:
                return temperatures['coretemp'][0].current
            elif 'cpu_thermal' in temperatures:
                return temperatures['cpu_thermal'][0].current
        except Exception:
            pass
        return None

    def _evaluate_fallback_need(self):
        """Evaluate if fallback is needed"""
        current_metrics = self.health_metrics.get(self.current_acceleration)

        if not current_metrics:
            return

        failure_threshold = self.config.get("failure_threshold", 3)

        # Check if current acceleration is failing
        if current_metrics.status in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
            self.failure_counts[self.current_acceleration] += 1

            if self.failure_counts[self.current_acceleration] >= failure_threshold:
                # Find next available acceleration
                next_acceleration = self._find_next_acceleration()
                if next_acceleration and next_acceleration != self.current_acceleration:
                    reason = f"Health status: {current_metrics.status.value}, failures: {self.failure_counts[self.current_acceleration]}"
                    self._activate_fallback(next_acceleration, reason)
        else:
            # Reset failure count on healthy status
            if current_metrics.status == HealthStatus.HEALTHY:
                self.failure_counts[self.current_acceleration] = 0

    def _find_next_acceleration(self) -> Optional[AccelerationType]:
        """Find next available acceleration type"""
        fallback_chain = sorted(
            self.config.get("fallback_chain", []),
            key=lambda x: x.get("priority", 999)
        )

        # Find current acceleration in chain
        current_priority = None
        for fallback_config in fallback_chain:
            try:
                accel_type = AccelerationType(fallback_config.get("type", ""))
                if accel_type == self.current_acceleration:
                    current_priority = fallback_config.get("priority", 999)
                    break
            except ValueError:
                continue

        # Find next available acceleration with higher priority number (lower priority)
        for fallback_config in fallback_chain:
            if not fallback_config.get("enabled", False):
                continue

            priority = fallback_config.get("priority", 999)
            if current_priority is not None and priority <= current_priority:
                continue

            try:
                accel_type = AccelerationType(fallback_config.get("type", ""))
                if self._test_acceleration_availability(accel_type):
                    metrics = self.health_metrics.get(accel_type)
                    if metrics and metrics.status not in [HealthStatus.CRITICAL, HealthStatus.FAILED]:
                        return accel_type
            except ValueError:
                continue

        return None

    def _activate_fallback(self, target_acceleration: AccelerationType, reason: str):
        """Activate fallback to target acceleration"""
        start_time = time.time()

        logger.warning(f"Activating fallback from {self.current_acceleration.value} to {target_acceleration.value}: {reason}")

        # Record fallback event
        previous_acceleration = self.current_acceleration
        self.current_acceleration = target_acceleration

        # Reset failure count for new acceleration
        self.failure_counts[target_acceleration] = 0

        recovery_time_ms = (time.time() - start_time) * 1000

        fallback_event = FallbackEvent(
            timestamp=time.time(),
            from_acceleration=previous_acceleration,
            to_acceleration=target_acceleration,
            reason=reason,
            recovery_time_ms=recovery_time_ms
        )

        self.fallback_events.append(fallback_event)

        # Keep only last 100 events
        if len(self.fallback_events) > 100:
            self.fallback_events = self.fallback_events[-100:]

        logger.info(f"Fallback activated successfully in {recovery_time_ms:.1f}ms")

    def export_health_report(self, output_path: Optional[str] = None) -> str:
        """Export health monitoring report"""
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"/var/log/military-tpm/health_report_{timestamp}.json"

        report = {
            "timestamp": time.time(),
            "current_acceleration": self.current_acceleration.value,
            "health_metrics": {
                accel.value: {
                    "status": metrics.status.value,
                    "response_time_ms": metrics.response_time_ms,
                    "throughput_ops_sec": metrics.throughput_ops_sec,
                    "error_rate_percent": metrics.error_rate_percent,
                    "temperature_celsius": metrics.temperature_celsius,
                    "utilization_percent": metrics.utilization_percent,
                    "last_checked": metrics.last_checked
                }
                for accel, metrics in self.health_metrics.items()
            },
            "failure_counts": {accel.value: count for accel, count in self.failure_counts.items()},
            "fallback_events": [
                {
                    "timestamp": event.timestamp,
                    "from": event.from_acceleration.value,
                    "to": event.to_acceleration.value,
                    "reason": event.reason,
                    "recovery_time_ms": event.recovery_time_ms
                }
                for event in self.fallback_events
            ],
            "configuration": self.config
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Health report exported: {output_path}")
        return output_path


def main():
    """Main health monitor entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="TPM2 Acceleration Health Monitor")
    parser.add_argument("--config", default="/etc/military-tpm/fallback.json",
                       help="Configuration file path")
    parser.add_argument("--export-report", action="store_true",
                       help="Export health report and exit")
    parser.add_argument("--status", action="store_true",
                       help="Show current status and exit")

    args = parser.parse_args()

    # Create health monitor
    monitor = AccelerationHealthMonitor(args.config)

    if args.export_report:
        # Export report and exit
        monitor.start()
        time.sleep(2)  # Allow some health checks
        report_path = monitor.export_health_report()
        print(f"Health report exported: {report_path}")
        monitor.stop()
        return

    if args.status:
        # Show status and exit
        monitor.start()
        time.sleep(2)  # Allow some health checks
        status = monitor.get_current_status()
        print(json.dumps(status, indent=2))
        monitor.stop()
        return

    # Run health monitor
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