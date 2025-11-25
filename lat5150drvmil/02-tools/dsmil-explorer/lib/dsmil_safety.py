#!/usr/bin/env python3
"""
DSMIL Safety Validation Library

Provides multi-layer safety checks and validation for DSMIL device operations.
Implements fail-safe mechanisms to prevent destructive operations.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import time
import subprocess
from typing import Dict, List, Tuple, Optional
from enum import Enum

class SafetyLevel(Enum):
    """Safety levels for device operations"""
    SAFE = "SAFE"           # Known safe operations (read-only)
    CAUTION = "CAUTION"     # Potentially safe, needs monitoring
    RISKY = "RISKY"         # Known risks, extra validation required
    DANGEROUS = "DANGEROUS" # High risk operations
    FORBIDDEN = "FORBIDDEN" # Permanently blocked

class DeviceRisk(Enum):
    """Risk categories for devices"""
    MONITORED = "MONITORED"     # Safe, already monitored
    UNKNOWN = "UNKNOWN"         # Unknown, needs exploration
    RISKY = "RISKY"            # Identified but risky
    QUARANTINED = "QUARANTINED" # Permanently blocked

# Quarantined devices - NEVER ACCESS
QUARANTINED_DEVICES = {
    0x8009: "DATA_DESTRUCTION",
    0x800A: "CASCADE_WIPE",
    0x800B: "HARDWARE_SANITIZE",
    0x8019: "NETWORK_KILL",
    0x8029: "COMMS_BLACKOUT",
}

# Known safe devices (read-only monitoring)
SAFE_DEVICES = {
    0x8003: "AUDIT_LOG",
    0x8004: "EVENT_LOGGER",
    0x8005: "PERFORMANCE_MONITOR",
    0x8006: "THERMAL_SENSOR",
    0x8007: "POWER_STATE",
    0x802A: "NETWORK_MONITOR",
}

# Risky devices (requires extra caution)
RISKY_DEVICES = {
    0x8000: "TPM_CONTROL",
    0x8001: "BOOT_SECURITY",
    0x8002: "CREDENTIAL_VAULT",
    0x8008: "EMERGENCY_RESPONSE",
    0x8013: "KEY_MANAGEMENT",
    0x8016: "VPN_CONTROLLER",
    0x8017: "REMOTE_ACCESS",
    0x8018: "PRE_ISOLATION",
}

class SafetyValidator:
    """Multi-layer safety validation for DSMIL operations"""

    def __init__(self, emergency_stop_callback=None):
        self.emergency_stop_callback = emergency_stop_callback
        self.operation_count = 0
        self.safety_violations = 0
        self.last_check_time = time.time()

    def check_device_access(self, device_id: int) -> Tuple[bool, str, SafetyLevel]:
        """
        Check if device access is allowed

        Returns:
            (allowed, reason, safety_level)
        """
        # Check quarantine list first
        if device_id in QUARANTINED_DEVICES:
            return (False,
                    f"QUARANTINED: {QUARANTINED_DEVICES[device_id]} - NEVER ACCESS",
                    SafetyLevel.FORBIDDEN)

        # Check if device is in safe list
        if device_id in SAFE_DEVICES:
            return (True,
                    f"Safe device: {SAFE_DEVICES[device_id]}",
                    SafetyLevel.SAFE)

        # Check if device is risky
        if device_id in RISKY_DEVICES:
            return (True,
                    f"Risky device: {RISKY_DEVICES[device_id]} - CAUTION REQUIRED",
                    SafetyLevel.RISKY)

        # Unknown device - allow with caution
        return (True,
                "Unknown device - exploration mode",
                SafetyLevel.CAUTION)

    def validate_operation(self, device_id: int, operation: str,
                          is_write: bool = False) -> Tuple[bool, str]:
        """
        Validate a specific operation on a device

        Args:
            device_id: Device to access
            operation: Operation description
            is_write: Whether operation involves writes

        Returns:
            (allowed, reason)
        """
        # Check device access first
        allowed, reason, level = self.check_device_access(device_id)

        if not allowed:
            self.safety_violations += 1
            return (False, reason)

        # Write operations require extra validation
        if is_write:
            if level == SafetyLevel.FORBIDDEN:
                self.safety_violations += 1
                return (False, f"Write blocked: {reason}")

            if level == SafetyLevel.RISKY:
                # Allow risky writes only with explicit confirmation
                return (False,
                        f"Write to risky device requires manual confirmation: {reason}")

        self.operation_count += 1
        return (True, f"Operation validated: {operation}")

    def check_system_health(self) -> Tuple[bool, Dict[str, any]]:
        """
        Check overall system health before operations

        Returns:
            (healthy, health_metrics)
        """
        health = {
            "timestamp": time.time(),
            "uptime": self._get_uptime(),
            "load_average": self._get_load_average(),
            "memory_available": self._get_memory_available(),
            "disk_space": self._get_disk_space(),
            "thermal_ok": self._check_thermal(),
        }

        # Check if system is healthy
        healthy = (
            health["load_average"] < 10.0 and
            health["memory_available"] > 100 * 1024 * 1024 and  # 100 MB
            health["disk_space"] > 1024 * 1024 * 1024 and  # 1 GB
            health["thermal_ok"]
        )

        return (healthy, health)

    def emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        print(f"\n{'='*80}")
        print(f"EMERGENCY STOP TRIGGERED: {reason}")
        print(f"{'='*80}\n")

        if self.emergency_stop_callback:
            self.emergency_stop_callback(reason)

    def get_device_risk_level(self, device_id: int) -> DeviceRisk:
        """Get risk level for a device"""
        if device_id in QUARANTINED_DEVICES:
            return DeviceRisk.QUARANTINED
        elif device_id in SAFE_DEVICES:
            return DeviceRisk.MONITORED
        elif device_id in RISKY_DEVICES:
            return DeviceRisk.RISKY
        else:
            return DeviceRisk.UNKNOWN

    # System health check helpers

    def _get_uptime(self) -> float:
        """Get system uptime in seconds"""
        try:
            with open('/proc/uptime', 'r') as f:
                return float(f.read().split()[0])
        except:
            return 0.0

    def _get_load_average(self) -> float:
        """Get 1-minute load average"""
        try:
            return os.getloadavg()[0]
        except:
            return 0.0

    def _get_memory_available(self) -> int:
        """Get available memory in bytes"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemAvailable:'):
                        return int(line.split()[1]) * 1024
        except:
            pass
        return 0

    def _get_disk_space(self) -> int:
        """Get available disk space in bytes"""
        try:
            stat = os.statvfs('/')
            return stat.f_bavail * stat.f_frsize
        except:
            return 0

    def _check_thermal(self) -> bool:
        """Check if system thermal is OK"""
        try:
            # Check thermal zones
            thermal_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/class/thermal/thermal_zone1/temp',
            ]

            for path in thermal_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = int(f.read().strip()) / 1000  # Convert to Celsius
                        if temp > 85:  # Critical threshold
                            return False

            return True
        except:
            return True  # Assume OK if can't read

def create_safety_validator() -> SafetyValidator:
    """Factory function to create a safety validator"""
    return SafetyValidator()

if __name__ == "__main__":
    # Self-test
    print("DSMIL Safety Validation Library - Self Test")
    print("=" * 80)

    validator = create_safety_validator()

    # Test device checks
    test_devices = [
        (0x8003, "Safe device"),
        (0x8009, "Quarantined device"),
        (0x8000, "Risky device"),
        (0x8030, "Unknown device"),
    ]

    for device_id, desc in test_devices:
        allowed, reason, level = validator.check_device_access(device_id)
        risk = validator.get_device_risk_level(device_id)
        print(f"\nDevice 0x{device_id:04X} ({desc}):")
        print(f"  Allowed: {allowed}")
        print(f"  Reason: {reason}")
        print(f"  Level: {level.value}")
        print(f"  Risk: {risk.value}")

    # Test system health
    print("\n" + "=" * 80)
    print("System Health Check:")
    healthy, metrics = validator.check_system_health()
    print(f"  Healthy: {healthy}")
    for key, value in metrics.items():
        if key == "timestamp":
            continue
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
    print(f"Safety Validator Statistics:")
    print(f"  Operations: {validator.operation_count}")
    print(f"  Violations: {validator.safety_violations}")
