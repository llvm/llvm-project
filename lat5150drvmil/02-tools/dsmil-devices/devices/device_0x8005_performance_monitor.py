#!/usr/bin/env python3
"""
Device 0x8005: Performance Monitor / TPM-HSM Interface Controller

Hardware performance monitoring and TPM/HSM coordination interface.
Provides real-time metrics, system health monitoring, and HSM integration.

Device ID: 0x8005
Group: 0 (Core Security)
Risk Level: SAFE (Read-only monitoring operations)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class PerformanceMetric:
    """Performance metric identifiers"""
    CPU_USAGE = 0
    MEMORY_USAGE = 1
    DISK_IO = 2
    NETWORK_IO = 3
    CRYPTO_OPS = 4
    HSM_STATUS = 5
    TPM_ACTIVITY = 6
    THERMAL = 7
    POWER_CONSUMPTION = 8


class PerformanceMonitorDevice(DSMILDeviceBase):
    """Performance Monitor / TPM-HSM Interface Controller (0x8005)"""

    # Register map
    REG_MONITOR_STATUS = 0x00
    REG_CPU_USAGE = 0x04
    REG_MEMORY_USAGE = 0x08
    REG_DISK_IO = 0x0C
    REG_NETWORK_IO = 0x10
    REG_CRYPTO_OPS = 0x14
    REG_HSM_STATUS = 0x18
    REG_TPM_ACTIVITY = 0x1C
    REG_THERMAL_STATUS = 0x20
    REG_POWER_STATUS = 0x24

    # Status bits
    STATUS_MONITORING_ACTIVE = 0x01
    STATUS_HSM_CONNECTED = 0x02
    STATUS_TPM_AVAILABLE = 0x04
    STATUS_METRICS_VALID = 0x08
    STATUS_THRESHOLD_EXCEEDED = 0x10
    STATUS_LOGGING_ENABLED = 0x20

    def __init__(self, device_id: int = 0x8005,
                 name: str = "Performance Monitor",
                 description: str = "Performance Monitoring and TPM/HSM Interface"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.monitoring_active = False
        self.sample_interval = 1.0  # seconds
        self.metrics_history = []
        self.max_history = 1000

        # Current metrics
        self.current_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_io": 0,
            "network_io": 0,
            "crypto_ops": 0,
            "hsm_status": "disconnected",
            "tpm_activity": 0,
            "thermal": 0.0,
            "power_consumption": 0.0,
        }

        # Register map
        self.register_map = {
            "MONITOR_STATUS": {
                "offset": self.REG_MONITOR_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Monitoring system status"
            },
            "CPU_USAGE": {
                "offset": self.REG_CPU_USAGE,
                "size": 4,
                "access": "RO",
                "description": "CPU usage percentage (x100)"
            },
            "MEMORY_USAGE": {
                "offset": self.REG_MEMORY_USAGE,
                "size": 4,
                "access": "RO",
                "description": "Memory usage percentage (x100)"
            },
            "DISK_IO": {
                "offset": self.REG_DISK_IO,
                "size": 4,
                "access": "RO",
                "description": "Disk I/O operations per second"
            },
            "NETWORK_IO": {
                "offset": self.REG_NETWORK_IO,
                "size": 4,
                "access": "RO",
                "description": "Network I/O bytes per second"
            },
            "CRYPTO_OPS": {
                "offset": self.REG_CRYPTO_OPS,
                "size": 4,
                "access": "RO",
                "description": "Cryptographic operations per second"
            },
            "HSM_STATUS": {
                "offset": self.REG_HSM_STATUS,
                "size": 4,
                "access": "RO",
                "description": "HSM connection status"
            },
            "TPM_ACTIVITY": {
                "offset": self.REG_TPM_ACTIVITY,
                "size": 4,
                "access": "RO",
                "description": "TPM operations per second"
            },
            "THERMAL_STATUS": {
                "offset": self.REG_THERMAL_STATUS,
                "size": 4,
                "access": "RO",
                "description": "System thermal status (°C x 100)"
            },
            "POWER_STATUS": {
                "offset": self.REG_POWER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Power consumption (mW)"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Performance Monitor device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize monitoring
            self.monitoring_active = True
            self._update_metrics()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "monitoring_active": self.monitoring_active,
                "sample_interval": self.sample_interval,
                "metrics_available": len(self.current_metrics),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_ONLY,
            DeviceCapability.STATUS_REPORTING,
            DeviceCapability.EVENT_MONITORING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        status_reg = self._read_status_register()

        return {
            "monitoring_active": bool(status_reg & self.STATUS_MONITORING_ACTIVE),
            "hsm_connected": bool(status_reg & self.STATUS_HSM_CONNECTED),
            "tpm_available": bool(status_reg & self.STATUS_TPM_AVAILABLE),
            "metrics_valid": bool(status_reg & self.STATUS_METRICS_VALID),
            "threshold_exceeded": bool(status_reg & self.STATUS_THRESHOLD_EXCEEDED),
            "logging_enabled": bool(status_reg & self.STATUS_LOGGING_ENABLED),
            "sample_interval": self.sample_interval,
            "history_samples": len(self.metrics_history),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            # Update metrics before reading
            self._update_metrics()

            if register == "MONITOR_STATUS":
                value = self._read_status_register()
            elif register == "CPU_USAGE":
                value = int(self.current_metrics["cpu_usage"] * 100)
            elif register == "MEMORY_USAGE":
                value = int(self.current_metrics["memory_usage"] * 100)
            elif register == "DISK_IO":
                value = self.current_metrics["disk_io"]
            elif register == "NETWORK_IO":
                value = self.current_metrics["network_io"]
            elif register == "CRYPTO_OPS":
                value = self.current_metrics["crypto_ops"]
            elif register == "HSM_STATUS":
                value = 1 if self.current_metrics["hsm_status"] == "connected" else 0
            elif register == "TPM_ACTIVITY":
                value = self.current_metrics["tpm_activity"]
            elif register == "THERMAL_STATUS":
                value = int(self.current_metrics["thermal"] * 100)
            elif register == "POWER_STATUS":
                value = int(self.current_metrics["power_consumption"])
            else:
                value = 0

            self._record_operation(True)
            return OperationResult(True, data={
                "register": register,
                "value": value,
                "hex": f"0x{value:08X}",
            })

        except Exception as e:
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    # Performance Monitor specific operations

    def get_current_metrics(self) -> OperationResult:
        """Get current performance metrics snapshot"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_metrics()

        self._record_operation(True)
        return OperationResult(True, data={
            "timestamp": time.time(),
            "metrics": self.current_metrics.copy(),
        })

    def get_metrics_summary(self) -> OperationResult:
        """Get summary of recent metrics"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if not self.metrics_history:
            return OperationResult(False, error="No metrics history available")

        # Calculate averages
        summary = {}
        for key in self.current_metrics.keys():
            if key == "hsm_status":
                continue

            values = [m[key] for m in self.metrics_history if key in m]
            if values:
                summary[key] = {
                    "current": values[-1] if values else 0,
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }

        self._record_operation(True)
        return OperationResult(True, data={
            "samples": len(self.metrics_history),
            "summary": summary,
        })

    def get_hsm_status(self) -> OperationResult:
        """Get HSM connection status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Simulated HSM status
        hsm_info = {
            "connected": self.current_metrics["hsm_status"] == "connected",
            "status": self.current_metrics["hsm_status"],
            "type": "Virtual HSM" if self.current_metrics["hsm_status"] == "connected" else None,
            "operations": 0,
        }

        self._record_operation(True)
        return OperationResult(True, data=hsm_info)

    def get_tpm_activity(self) -> OperationResult:
        """Get TPM activity metrics"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        tpm_info = {
            "available": self.current_metrics["tpm_activity"] >= 0,
            "operations_per_sec": self.current_metrics["tpm_activity"],
            "total_operations": self.operation_count,
        }

        self._record_operation(True)
        return OperationResult(True, data=tpm_info)

    def get_thermal_status(self) -> OperationResult:
        """Get thermal status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        thermal = self.current_metrics["thermal"]

        # Thermal thresholds
        status = "normal"
        if thermal > 85:
            status = "critical"
        elif thermal > 75:
            status = "warning"
        elif thermal > 65:
            status = "elevated"

        self._record_operation(True)
        return OperationResult(True, data={
            "temperature_celsius": thermal,
            "status": status,
            "critical_threshold": 85,
            "warning_threshold": 75,
        })

    def get_crypto_performance(self) -> OperationResult:
        """Get cryptographic operation performance"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        crypto_info = {
            "operations_per_sec": self.current_metrics["crypto_ops"],
            "throughput_estimate": f"{self.current_metrics['crypto_ops'] * 2048} bytes/sec",
        }

        self._record_operation(True)
        return OperationResult(True, data=crypto_info)

    def start_monitoring(self, interval: float = 1.0) -> OperationResult:
        """Start continuous monitoring"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self.monitoring_active = True
        self.sample_interval = interval

        self._record_operation(True)
        return OperationResult(True, data={
            "monitoring": True,
            "interval": interval,
        })

    def stop_monitoring(self) -> OperationResult:
        """Stop continuous monitoring"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self.monitoring_active = False

        self._record_operation(True)
        return OperationResult(True, data={
            "monitoring": False,
            "samples_collected": len(self.metrics_history),
        })

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read monitor status register (simulated)"""
        status = self.STATUS_MONITORING_ACTIVE | self.STATUS_METRICS_VALID

        if self.current_metrics["hsm_status"] == "connected":
            status |= self.STATUS_HSM_CONNECTED

        if self.current_metrics["tpm_activity"] >= 0:
            status |= self.STATUS_TPM_AVAILABLE

        if self.current_metrics["cpu_usage"] > 80 or self.current_metrics["thermal"] > 75:
            status |= self.STATUS_THRESHOLD_EXCEEDED

        status |= self.STATUS_LOGGING_ENABLED

        return status

    def _update_metrics(self):
        """Update current metrics (simulated)"""
        import random

        # Simulate realistic metrics
        self.current_metrics = {
            "cpu_usage": 15.0 + random.random() * 25.0,  # 15-40%
            "memory_usage": 45.0 + random.random() * 15.0,  # 45-60%
            "disk_io": int(100 + random.random() * 200),  # 100-300 ops/s
            "network_io": int(1000 + random.random() * 5000),  # 1-6 KB/s
            "crypto_ops": int(50 + random.random() * 100),  # 50-150 ops/s
            "hsm_status": "disconnected",  # Simulated as disconnected
            "tpm_activity": int(10 + random.random() * 40),  # 10-50 ops/s
            "thermal": 55.0 + random.random() * 15.0,  # 55-70°C
            "power_consumption": 8000 + random.random() * 4000,  # 8-12W
        }

        # Store in history
        metrics_snapshot = self.current_metrics.copy()
        metrics_snapshot["timestamp"] = time.time()
        self.metrics_history.append(metrics_snapshot)

        # Trim history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
