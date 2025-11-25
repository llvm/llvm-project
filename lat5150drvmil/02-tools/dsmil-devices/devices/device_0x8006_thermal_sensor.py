#!/usr/bin/env python3
"""
Device 0x8006: Thermal Sensor

System thermal monitoring and temperature management for all platform components.
Monitors CPU, GPU, chipset, and sensor array temperatures with threshold alerts.

Device ID: 0x8006
Group: 0 (Core Security)
Risk Level: SAFE (Read-only thermal monitoring)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import time
import random

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class ThermalZone:
    """Thermal zone identifiers"""
    CPU = 0
    GPU = 1
    CHIPSET = 2
    MEMORY = 3
    STORAGE = 4
    SENSOR_ARRAY = 5
    AMBIENT = 6
    VRM = 7  # Voltage Regulator Module


class ThermalStatus:
    """Thermal status levels"""
    NORMAL = 0
    WARM = 1
    HOT = 2
    CRITICAL = 3
    EMERGENCY = 4


class ThermalSensorDevice(DSMILDeviceBase):
    """Thermal Sensor (0x8006)"""

    # Register map
    REG_THERMAL_STATUS = 0x00
    REG_CPU_TEMP = 0x04
    REG_GPU_TEMP = 0x08
    REG_CHIPSET_TEMP = 0x0C
    REG_AMBIENT_TEMP = 0x10
    REG_MAX_TEMP = 0x14
    REG_ALERT_COUNT = 0x18
    REG_THROTTLE_STATUS = 0x1C

    # Status bits
    STATUS_MONITORING_ACTIVE = 0x01
    STATUS_ALERT_ACTIVE = 0x02
    STATUS_CRITICAL_TEMP = 0x04
    STATUS_THROTTLING = 0x08
    STATUS_FAN_CONTROL = 0x10
    STATUS_EMERGENCY = 0x20

    # Temperature thresholds (Celsius)
    THRESHOLD_WARM = 60.0
    THRESHOLD_HOT = 75.0
    THRESHOLD_CRITICAL = 90.0
    THRESHOLD_EMERGENCY = 100.0

    def __init__(self, device_id: int = 0x8006,
                 name: str = "Thermal Sensor",
                 description: str = "System Thermal Monitoring and Management"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.thermal_zones = {}
        self.alert_history = []
        self.monitoring_active = True
        self.fan_control_enabled = True
        self.throttling_active = False
        self.max_temp_recorded = 0.0

        # Initialize thermal zones
        self._initialize_thermal_zones()

        # Register map
        self.register_map = {
            "THERMAL_STATUS": {
                "offset": self.REG_THERMAL_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Thermal monitoring status"
            },
            "CPU_TEMP": {
                "offset": self.REG_CPU_TEMP,
                "size": 4,
                "access": "RO",
                "description": "CPU temperature (°C * 100)"
            },
            "GPU_TEMP": {
                "offset": self.REG_GPU_TEMP,
                "size": 4,
                "access": "RO",
                "description": "GPU temperature (°C * 100)"
            },
            "CHIPSET_TEMP": {
                "offset": self.REG_CHIPSET_TEMP,
                "size": 4,
                "access": "RO",
                "description": "Chipset temperature (°C * 100)"
            },
            "AMBIENT_TEMP": {
                "offset": self.REG_AMBIENT_TEMP,
                "size": 4,
                "access": "RO",
                "description": "Ambient temperature (°C * 100)"
            },
            "MAX_TEMP": {
                "offset": self.REG_MAX_TEMP,
                "size": 4,
                "access": "RO",
                "description": "Maximum recorded temperature (°C * 100)"
            },
            "ALERT_COUNT": {
                "offset": self.REG_ALERT_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of thermal alerts"
            },
            "THROTTLE_STATUS": {
                "offset": self.REG_THROTTLE_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Thermal throttling status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Thermal Sensor device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize thermal zones
            self._initialize_thermal_zones()
            self.monitoring_active = True
            self.fan_control_enabled = True
            self.throttling_active = False
            self.alert_history = []

            # Update temperatures
            self._update_temperatures()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "monitoring_active": self.monitoring_active,
                "thermal_zones": len(self.thermal_zones),
                "fan_control": self.fan_control_enabled,
                "max_temp": self.max_temp_recorded,
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
            "alert_active": bool(status_reg & self.STATUS_ALERT_ACTIVE),
            "critical_temp": bool(status_reg & self.STATUS_CRITICAL_TEMP),
            "throttling": bool(status_reg & self.STATUS_THROTTLING),
            "fan_control": bool(status_reg & self.STATUS_FAN_CONTROL),
            "emergency": bool(status_reg & self.STATUS_EMERGENCY),
            "max_temp": self.max_temp_recorded,
            "alert_count": len(self.alert_history),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "THERMAL_STATUS":
                value = self._read_status_register()
            elif register == "CPU_TEMP":
                value = int(self.thermal_zones[ThermalZone.CPU]['temperature'] * 100)
            elif register == "GPU_TEMP":
                value = int(self.thermal_zones[ThermalZone.GPU]['temperature'] * 100)
            elif register == "CHIPSET_TEMP":
                value = int(self.thermal_zones[ThermalZone.CHIPSET]['temperature'] * 100)
            elif register == "AMBIENT_TEMP":
                value = int(self.thermal_zones[ThermalZone.AMBIENT]['temperature'] * 100)
            elif register == "MAX_TEMP":
                value = int(self.max_temp_recorded * 100)
            elif register == "ALERT_COUNT":
                value = len(self.alert_history)
            elif register == "THROTTLE_STATUS":
                value = 1 if self.throttling_active else 0
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

    # Thermal Sensor specific operations

    def get_all_temperatures(self) -> OperationResult:
        """Get all thermal zone temperatures"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_temperatures()

        temps = {}
        for zone_id, zone_info in self.thermal_zones.items():
            zone_name = self._get_zone_name(zone_id)
            temps[zone_name] = {
                "temperature": zone_info['temperature'],
                "unit": zone_info['unit'],
                "status": self._get_thermal_status_name(
                    self._assess_temperature(zone_info['temperature'])
                ),
            }

        self._record_operation(True)
        return OperationResult(True, data={
            "temperatures": temps,
            "timestamp": time.time(),
        })

    def get_zone_temperature(self, zone: int) -> OperationResult:
        """Get specific thermal zone temperature"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if zone not in self.thermal_zones:
            return OperationResult(False, error=f"Unknown thermal zone: {zone}")

        self._update_temperatures()

        zone_info = self.thermal_zones[zone]
        temp = zone_info['temperature']

        self._record_operation(True)
        return OperationResult(True, data={
            "zone": self._get_zone_name(zone),
            "temperature": temp,
            "unit": zone_info['unit'],
            "status": self._get_thermal_status_name(self._assess_temperature(temp)),
            "threshold_warm": self.THRESHOLD_WARM,
            "threshold_hot": self.THRESHOLD_HOT,
            "threshold_critical": self.THRESHOLD_CRITICAL,
        })

    def get_thermal_summary(self) -> OperationResult:
        """Get thermal system summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_temperatures()

        # Find hottest zone
        hottest_temp = 0.0
        hottest_zone = None
        for zone_id, zone_info in self.thermal_zones.items():
            if zone_info['temperature'] > hottest_temp:
                hottest_temp = zone_info['temperature']
                hottest_zone = zone_id

        # Count zones by status
        status_counts = {}
        for zone_info in self.thermal_zones.values():
            status = self._assess_temperature(zone_info['temperature'])
            status_name = self._get_thermal_status_name(status)
            status_counts[status_name] = status_counts.get(status_name, 0) + 1

        summary = {
            "overall_status": self._get_overall_status(),
            "hottest_zone": self._get_zone_name(hottest_zone) if hottest_zone else "None",
            "hottest_temp": hottest_temp,
            "max_temp_recorded": self.max_temp_recorded,
            "zones_by_status": status_counts,
            "throttling_active": self.throttling_active,
            "alert_count": len(self.alert_history),
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_alert_history(self, limit: int = 50) -> OperationResult:
        """Get thermal alert history"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        alerts = self.alert_history[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "alerts": alerts,
            "total": len(alerts),
            "showing": f"Last {min(limit, len(self.alert_history))} of {len(self.alert_history)}",
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get thermal sensor statistics"""
        stats = super().get_statistics()

        self._update_temperatures()

        stats.update({
            "thermal_zones": len(self.thermal_zones),
            "max_temp_recorded": self.max_temp_recorded,
            "alert_count": len(self.alert_history),
            "throttling_active": self.throttling_active,
            "monitoring_active": self.monitoring_active,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read thermal status register (simulated)"""
        status = self.STATUS_MONITORING_ACTIVE if self.monitoring_active else 0

        # Check for alerts
        for zone_info in self.thermal_zones.values():
            temp_status = self._assess_temperature(zone_info['temperature'])
            if temp_status >= ThermalStatus.HOT:
                status |= self.STATUS_ALERT_ACTIVE
            if temp_status >= ThermalStatus.CRITICAL:
                status |= self.STATUS_CRITICAL_TEMP
            if temp_status >= ThermalStatus.EMERGENCY:
                status |= self.STATUS_EMERGENCY

        if self.throttling_active:
            status |= self.STATUS_THROTTLING

        if self.fan_control_enabled:
            status |= self.STATUS_FAN_CONTROL

        return status

    def _initialize_thermal_zones(self):
        """Initialize thermal zones with baseline temperatures"""
        self.thermal_zones = {
            ThermalZone.CPU: {
                "name": "CPU",
                "temperature": 45.0,
                "unit": "°C",
                "baseline": 45.0,
            },
            ThermalZone.GPU: {
                "name": "GPU",
                "temperature": 42.0,
                "unit": "°C",
                "baseline": 42.0,
            },
            ThermalZone.CHIPSET: {
                "name": "Chipset",
                "temperature": 50.0,
                "unit": "°C",
                "baseline": 50.0,
            },
            ThermalZone.MEMORY: {
                "name": "Memory",
                "temperature": 38.0,
                "unit": "°C",
                "baseline": 38.0,
            },
            ThermalZone.STORAGE: {
                "name": "Storage",
                "temperature": 35.0,
                "unit": "°C",
                "baseline": 35.0,
            },
            ThermalZone.SENSOR_ARRAY: {
                "name": "Sensor Array",
                "temperature": 40.0,
                "unit": "°C",
                "baseline": 40.0,
            },
            ThermalZone.AMBIENT: {
                "name": "Ambient",
                "temperature": 22.0,
                "unit": "°C",
                "baseline": 22.0,
            },
            ThermalZone.VRM: {
                "name": "VRM",
                "temperature": 55.0,
                "unit": "°C",
                "baseline": 55.0,
            },
        }
        self.max_temp_recorded = 55.0

    def _update_temperatures(self):
        """Update thermal zone temperatures (simulated)"""
        for zone_id, zone_info in self.thermal_zones.items():
            # Simulate slight temperature variation
            baseline = zone_info['baseline']
            variation = random.uniform(-2.0, 3.0)
            new_temp = baseline + variation

            zone_info['temperature'] = round(new_temp, 1)

            # Update max temp
            if new_temp > self.max_temp_recorded:
                self.max_temp_recorded = round(new_temp, 1)

    def _assess_temperature(self, temp: float) -> int:
        """Assess temperature status"""
        if temp >= self.THRESHOLD_EMERGENCY:
            return ThermalStatus.EMERGENCY
        elif temp >= self.THRESHOLD_CRITICAL:
            return ThermalStatus.CRITICAL
        elif temp >= self.THRESHOLD_HOT:
            return ThermalStatus.HOT
        elif temp >= self.THRESHOLD_WARM:
            return ThermalStatus.WARM
        else:
            return ThermalStatus.NORMAL

    def _get_overall_status(self) -> str:
        """Get overall thermal status"""
        max_status = ThermalStatus.NORMAL
        for zone_info in self.thermal_zones.values():
            status = self._assess_temperature(zone_info['temperature'])
            if status > max_status:
                max_status = status
        return self._get_thermal_status_name(max_status)

    def _get_zone_name(self, zone: int) -> str:
        """Get thermal zone name"""
        names = {
            ThermalZone.CPU: "CPU",
            ThermalZone.GPU: "GPU",
            ThermalZone.CHIPSET: "Chipset",
            ThermalZone.MEMORY: "Memory",
            ThermalZone.STORAGE: "Storage",
            ThermalZone.SENSOR_ARRAY: "Sensor Array",
            ThermalZone.AMBIENT: "Ambient",
            ThermalZone.VRM: "VRM",
        }
        return names.get(zone, "Unknown")

    def _get_thermal_status_name(self, status: int) -> str:
        """Get thermal status name"""
        names = {
            ThermalStatus.NORMAL: "Normal",
            ThermalStatus.WARM: "Warm",
            ThermalStatus.HOT: "Hot",
            ThermalStatus.CRITICAL: "Critical",
            ThermalStatus.EMERGENCY: "Emergency",
        }
        return names.get(status, "Unknown")


def main():
    """Test Thermal Sensor device"""
    print("=" * 80)
    print("Device 0x8006: Thermal Sensor - Test")
    print("=" * 80)

    device = ThermalSensorDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Thermal zones: {result.data['thermal_zones']}")
        print(f"   Max temp: {result.data['max_temp']:.1f}°C")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   Monitoring active: {status['monitoring_active']}")
    print(f"   Alert active: {status['alert_active']}")
    print(f"   Max temp: {status['max_temp']:.1f}°C")

    # Get all temperatures
    print("\n3. Getting all temperatures...")
    result = device.get_all_temperatures()
    if result.success:
        for zone, info in result.data['temperatures'].items():
            print(f"   {zone}: {info['temperature']:.1f}{info['unit']} ({info['status']})")

    # Get CPU temperature
    print("\n4. Getting CPU temperature...")
    result = device.get_zone_temperature(ThermalZone.CPU)
    if result.success:
        print(f"   Zone: {result.data['zone']}")
        print(f"   Temperature: {result.data['temperature']:.1f}{result.data['unit']}")
        print(f"   Status: {result.data['status']}")

    # Get thermal summary
    print("\n5. Getting thermal summary...")
    result = device.get_thermal_summary()
    if result.success:
        print(f"   Overall status: {result.data['overall_status']}")
        print(f"   Hottest zone: {result.data['hottest_zone']}")
        print(f"   Hottest temp: {result.data['hottest_temp']:.1f}°C")
        print(f"   Zones by status: {result.data['zones_by_status']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
