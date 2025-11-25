#!/usr/bin/env python3
"""
Device 0x8007: Power State / Power Management Controller

Advanced power management and ACPI state control for military systems.
Manages system power states, CPU power states, and power consumption optimization.

Device ID: 0x8007
Group: 0 (Core Security)
Risk Level: MONITORED (Power state changes are logged)

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


class SystemPowerState:
    """ACPI System Power States (S-states)"""
    S0_WORKING = 0          # Fully operational
    S1_SLEEP = 1            # Low wake latency sleep
    S3_SUSPEND_TO_RAM = 3   # Suspend to RAM
    S4_HIBERNATE = 4        # Suspend to disk
    S5_SOFT_OFF = 5         # Soft off


class ProcessorPowerState:
    """ACPI Processor Power States (C-states)"""
    C0_ACTIVE = 0           # CPU active
    C1_HALT = 1             # CPU halted
    C2_STOP_CLOCK = 2       # Stop CPU clock
    C3_SLEEP = 3            # Deep sleep


class PerformanceState:
    """ACPI Performance States (P-states)"""
    P0_MAX_PERFORMANCE = 0  # Maximum performance
    P1_HIGH = 1             # High performance
    P2_MEDIUM = 2           # Medium performance
    P3_LOW = 3              # Low performance
    P4_MIN = 4              # Minimum performance


class PowerPolicy:
    """Power management policies"""
    MAXIMUM_PERFORMANCE = 0
    BALANCED = 1
    POWER_SAVER = 2
    TACTICAL = 3            # Mission-optimized


class PowerStateDevice(DSMILDeviceBase):
    """Power State / Power Management Controller (0x8007)"""

    # Register map
    REG_POWER_STATUS = 0x00
    REG_SYSTEM_STATE = 0x04
    REG_PROCESSOR_STATE = 0x08
    REG_PERFORMANCE_STATE = 0x0C
    REG_POWER_CONSUMPTION = 0x10
    REG_BATTERY_STATUS = 0x14
    REG_POLICY = 0x18
    REG_WAKE_EVENTS = 0x1C

    # Status bits
    STATUS_PM_ACTIVE = 0x01
    STATUS_BATTERY_PRESENT = 0x02
    STATUS_AC_POWER = 0x04
    STATUS_LOW_POWER_MODE = 0x08
    STATUS_THERMAL_THROTTLE = 0x10
    STATUS_WAKE_ENABLED = 0x20

    def __init__(self, device_id: int = 0x8007,
                 name: str = "Power State Controller",
                 description: str = "Power Management and ACPI State Control"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.system_state = SystemPowerState.S0_WORKING
        self.processor_state = ProcessorPowerState.C0_ACTIVE
        self.performance_state = PerformanceState.P0_MAX_PERFORMANCE
        self.power_policy = PowerPolicy.BALANCED

        self.battery_present = True
        self.ac_power = True
        self.battery_level = 85.0
        self.power_consumption = 45.0  # Watts

        self.wake_events_enabled = True
        self.low_power_mode = False
        self.thermal_throttle = False

        self.state_transitions = []
        self.max_transitions = 1000

        # Register map
        self.register_map = {
            "POWER_STATUS": {
                "offset": self.REG_POWER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Power management status"
            },
            "SYSTEM_STATE": {
                "offset": self.REG_SYSTEM_STATE,
                "size": 4,
                "access": "RO",
                "description": "Current system power state (S-state)"
            },
            "PROCESSOR_STATE": {
                "offset": self.REG_PROCESSOR_STATE,
                "size": 4,
                "access": "RO",
                "description": "Current processor power state (C-state)"
            },
            "PERFORMANCE_STATE": {
                "offset": self.REG_PERFORMANCE_STATE,
                "size": 4,
                "access": "RO",
                "description": "Current performance state (P-state)"
            },
            "POWER_CONSUMPTION": {
                "offset": self.REG_POWER_CONSUMPTION,
                "size": 4,
                "access": "RO",
                "description": "Current power consumption (mW)"
            },
            "BATTERY_STATUS": {
                "offset": self.REG_BATTERY_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Battery level (percentage)"
            },
            "POLICY": {
                "offset": self.REG_POLICY,
                "size": 4,
                "access": "RO",
                "description": "Active power policy"
            },
            "WAKE_EVENTS": {
                "offset": self.REG_WAKE_EVENTS,
                "size": 4,
                "access": "RO",
                "description": "Wake event configuration"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Power State device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize power management
            self.system_state = SystemPowerState.S0_WORKING
            self.processor_state = ProcessorPowerState.C0_ACTIVE
            self.performance_state = PerformanceState.P0_MAX_PERFORMANCE
            self.power_policy = PowerPolicy.BALANCED

            self.battery_present = True
            self.ac_power = True
            self.battery_level = 85.0
            self.power_consumption = 45.0

            self.wake_events_enabled = True
            self.low_power_mode = False
            self.thermal_throttle = False

            self.state_transitions = []

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "system_state": self._get_system_state_name(self.system_state),
                "power_policy": self._get_policy_name(self.power_policy),
                "battery_present": self.battery_present,
                "ac_power": self.ac_power,
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
            "pm_active": bool(status_reg & self.STATUS_PM_ACTIVE),
            "battery_present": bool(status_reg & self.STATUS_BATTERY_PRESENT),
            "ac_power": bool(status_reg & self.STATUS_AC_POWER),
            "low_power_mode": bool(status_reg & self.STATUS_LOW_POWER_MODE),
            "thermal_throttle": bool(status_reg & self.STATUS_THERMAL_THROTTLE),
            "wake_enabled": bool(status_reg & self.STATUS_WAKE_ENABLED),
            "system_state": self._get_system_state_name(self.system_state),
            "battery_level": self.battery_level,
            "power_consumption": self.power_consumption,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "POWER_STATUS":
                value = self._read_status_register()
            elif register == "SYSTEM_STATE":
                value = self.system_state
            elif register == "PROCESSOR_STATE":
                value = self.processor_state
            elif register == "PERFORMANCE_STATE":
                value = self.performance_state
            elif register == "POWER_CONSUMPTION":
                value = int(self.power_consumption * 1000)  # Convert to mW
            elif register == "BATTERY_STATUS":
                value = int(self.battery_level)
            elif register == "POLICY":
                value = self.power_policy
            elif register == "WAKE_EVENTS":
                value = 1 if self.wake_events_enabled else 0
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

    # Power State specific operations

    def get_power_summary(self) -> OperationResult:
        """Get comprehensive power status summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        self._update_power_metrics()

        summary = {
            "system_state": self._get_system_state_name(self.system_state),
            "processor_state": self._get_processor_state_name(self.processor_state),
            "performance_state": self._get_performance_state_name(self.performance_state),
            "power_policy": self._get_policy_name(self.power_policy),
            "power_consumption": {
                "watts": self.power_consumption,
                "status": self._get_power_status(),
            },
            "battery": {
                "present": self.battery_present,
                "level": self.battery_level,
                "charging": self.ac_power and self.battery_level < 100,
            },
            "ac_power": self.ac_power,
            "low_power_mode": self.low_power_mode,
            "thermal_throttle": self.thermal_throttle,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_battery_info(self) -> OperationResult:
        """Get detailed battery information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        if not self.battery_present:
            return OperationResult(False, error="No battery present")

        self._update_power_metrics()

        # Estimate time remaining (simplified)
        time_remaining = None
        if not self.ac_power and self.power_consumption > 0:
            # Assume 50Wh battery capacity
            capacity_wh = 50.0
            remaining_wh = capacity_wh * (self.battery_level / 100.0)
            time_remaining = int((remaining_wh / self.power_consumption) * 60)  # minutes

        battery_info = {
            "present": self.battery_present,
            "level": self.battery_level,
            "charging": self.ac_power and self.battery_level < 100,
            "time_remaining_minutes": time_remaining,
            "health": "Good" if self.battery_level > 20 else "Low",
        }

        self._record_operation(True)
        return OperationResult(True, data=battery_info)

    def get_state_transitions(self, limit: int = 50) -> OperationResult:
        """Get recent power state transitions"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        transitions = self.state_transitions[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "transitions": transitions,
            "total": len(transitions),
            "showing": f"Last {min(limit, len(self.state_transitions))} of {len(self.state_transitions)}",
        })

    def get_performance_info(self) -> OperationResult:
        """Get CPU performance state information"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        perf_info = {
            "current_p_state": self._get_performance_state_name(self.performance_state),
            "current_c_state": self._get_processor_state_name(self.processor_state),
            "thermal_throttle": self.thermal_throttle,
            "power_limit": self._get_power_limit(),
        }

        self._record_operation(True)
        return OperationResult(True, data=perf_info)

    def get_statistics(self) -> Dict[str, Any]:
        """Get power management statistics"""
        stats = super().get_statistics()

        self._update_power_metrics()

        stats.update({
            "system_state": self._get_system_state_name(self.system_state),
            "power_consumption_watts": self.power_consumption,
            "battery_level": self.battery_level if self.battery_present else None,
            "ac_power": self.ac_power,
            "state_transitions": len(self.state_transitions),
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read power status register (simulated)"""
        status = self.STATUS_PM_ACTIVE

        if self.battery_present:
            status |= self.STATUS_BATTERY_PRESENT

        if self.ac_power:
            status |= self.STATUS_AC_POWER

        if self.low_power_mode:
            status |= self.STATUS_LOW_POWER_MODE

        if self.thermal_throttle:
            status |= self.STATUS_THERMAL_THROTTLE

        if self.wake_events_enabled:
            status |= self.STATUS_WAKE_ENABLED

        return status

    def _update_power_metrics(self):
        """Update power consumption metrics (simulated)"""
        # Simulate power consumption based on performance state
        base_power = {
            PerformanceState.P0_MAX_PERFORMANCE: 45.0,
            PerformanceState.P1_HIGH: 35.0,
            PerformanceState.P2_MEDIUM: 25.0,
            PerformanceState.P3_LOW: 15.0,
            PerformanceState.P4_MIN: 10.0,
        }

        self.power_consumption = base_power.get(self.performance_state, 30.0)
        self.power_consumption += random.uniform(-2.0, 2.0)

        # Update battery level if on battery
        if self.battery_present and not self.ac_power:
            # Simulate battery drain (very simplified)
            drain_rate = 0.1  # % per update
            self.battery_level = max(0.0, self.battery_level - drain_rate)

    def _get_power_status(self) -> str:
        """Get power consumption status"""
        if self.power_consumption > 50:
            return "High"
        elif self.power_consumption > 30:
            return "Normal"
        else:
            return "Low"

    def _get_power_limit(self) -> str:
        """Get power limit description"""
        if self.thermal_throttle:
            return "Thermal Limited"
        elif self.low_power_mode:
            return "Power Saver Limited"
        else:
            return "Unrestricted"

    def _get_system_state_name(self, state: int) -> str:
        """Get system power state name"""
        names = {
            SystemPowerState.S0_WORKING: "S0 (Working)",
            SystemPowerState.S1_SLEEP: "S1 (Sleep)",
            SystemPowerState.S3_SUSPEND_TO_RAM: "S3 (Suspend to RAM)",
            SystemPowerState.S4_HIBERNATE: "S4 (Hibernate)",
            SystemPowerState.S5_SOFT_OFF: "S5 (Soft Off)",
        }
        return names.get(state, "Unknown")

    def _get_processor_state_name(self, state: int) -> str:
        """Get processor power state name"""
        names = {
            ProcessorPowerState.C0_ACTIVE: "C0 (Active)",
            ProcessorPowerState.C1_HALT: "C1 (Halt)",
            ProcessorPowerState.C2_STOP_CLOCK: "C2 (Stop Clock)",
            ProcessorPowerState.C3_SLEEP: "C3 (Sleep)",
        }
        return names.get(state, "Unknown")

    def _get_performance_state_name(self, state: int) -> str:
        """Get performance state name"""
        names = {
            PerformanceState.P0_MAX_PERFORMANCE: "P0 (Max Performance)",
            PerformanceState.P1_HIGH: "P1 (High)",
            PerformanceState.P2_MEDIUM: "P2 (Medium)",
            PerformanceState.P3_LOW: "P3 (Low)",
            PerformanceState.P4_MIN: "P4 (Min)",
        }
        return names.get(state, "Unknown")

    def _get_policy_name(self, policy: int) -> str:
        """Get power policy name"""
        names = {
            PowerPolicy.MAXIMUM_PERFORMANCE: "Maximum Performance",
            PowerPolicy.BALANCED: "Balanced",
            PowerPolicy.POWER_SAVER: "Power Saver",
            PowerPolicy.TACTICAL: "Tactical",
        }
        return names.get(policy, "Unknown")


def main():
    """Test Power State device"""
    print("=" * 80)
    print("Device 0x8007: Power State Controller - Test")
    print("=" * 80)

    device = PowerStateDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   System state: {result.data['system_state']}")
        print(f"   Power policy: {result.data['power_policy']}")
        print(f"   Battery present: {result.data['battery_present']}")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   PM active: {status['pm_active']}")
    print(f"   System state: {status['system_state']}")
    print(f"   Battery level: {status['battery_level']:.1f}%")
    print(f"   Power consumption: {status['power_consumption']:.1f}W")

    # Get power summary
    print("\n3. Getting power summary...")
    result = device.get_power_summary()
    if result.success:
        print(f"   System state: {result.data['system_state']}")
        print(f"   Performance state: {result.data['performance_state']}")
        print(f"   Power consumption: {result.data['power_consumption']['watts']:.1f}W")
        print(f"   Battery level: {result.data['battery']['level']:.1f}%")

    # Get battery info
    print("\n4. Getting battery info...")
    result = device.get_battery_info()
    if result.success:
        print(f"   Battery level: {result.data['level']:.1f}%")
        print(f"   Charging: {result.data['charging']}")
        print(f"   Health: {result.data['health']}")

    # Get performance info
    print("\n5. Getting performance info...")
    result = device.get_performance_info()
    if result.success:
        print(f"   P-state: {result.data['current_p_state']}")
        print(f"   C-state: {result.data['current_c_state']}")
        print(f"   Power limit: {result.data['power_limit']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
