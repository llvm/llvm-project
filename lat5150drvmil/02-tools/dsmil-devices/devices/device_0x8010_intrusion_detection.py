#!/usr/bin/env python3
"""
Device 0x8010: Intrusion Detection System

Hardware-level intrusion detection and physical security monitoring.
Tracks chassis intrusion, port tampering, and physical access attempts.

Device ID: 0x8010
Group: 1 (Extended Security)
Risk Level: SAFE (READ operations are safe)

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


class IntrusionEventType(object):
    """Intrusion event types"""
    CHASSIS_OPENED = 0
    CHASSIS_CLOSED = 1
    PORT_ACCESSED = 2
    SEAL_BROKEN = 3
    SENSOR_TRIGGERED = 4
    TAMPER_DETECTED = 5


class IntrusionDetectionDevice(DSMILDeviceBase):
    """Intrusion Detection System (0x8010)"""

    # Register map
    REG_IDS_STATUS = 0x00
    REG_EVENT_COUNT = 0x04
    REG_LAST_EVENT = 0x08
    REG_SENSOR_STATUS = 0x0C
    REG_ALERT_STATUS = 0x10
    REG_CHASSIS_STATUS = 0x14

    # Status bits
    STATUS_ARMED = 0x01
    STATUS_INTRUSION_DETECTED = 0x02
    STATUS_CHASSIS_SECURE = 0x04
    STATUS_SEALS_INTACT = 0x08
    STATUS_ALL_SENSORS_OK = 0x10

    def __init__(self, device_id: int = 0x8010,
                 name: str = "Intrusion Detection",
                 description: str = "Physical Intrusion Detection System"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.armed = True
        self.intrusion_events = []
        self.sensor_states = {
            "chassis": True,  # True = secure
            "seal_1": True,
            "seal_2": True,
            "usb_ports": True,
            "ethernet_port": True,
        }

        # Register map
        self.register_map = {
            "IDS_STATUS": {
                "offset": self.REG_IDS_STATUS,
                "size": 4,
                "access": "RO",
                "description": "IDS status register"
            },
            "EVENT_COUNT": {
                "offset": self.REG_EVENT_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total intrusion events"
            },
            "LAST_EVENT": {
                "offset": self.REG_LAST_EVENT,
                "size": 4,
                "access": "RO",
                "description": "Last event type"
            },
            "SENSOR_STATUS": {
                "offset": self.REG_SENSOR_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Sensor status bitmap"
            },
            "ALERT_STATUS": {
                "offset": self.REG_ALERT_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Active alerts"
            },
            "CHASSIS_STATUS": {
                "offset": self.REG_CHASSIS_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Chassis security status"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize IDS device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize all sensors as secure
            for sensor in self.sensor_states:
                self.sensor_states[sensor] = True

            self.armed = True
            self.intrusion_events = []

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "armed": self.armed,
                "sensors": len(self.sensor_states),
                "all_secure": all(self.sensor_states.values()),
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
            "armed": bool(status_reg & self.STATUS_ARMED),
            "intrusion_detected": bool(status_reg & self.STATUS_INTRUSION_DETECTED),
            "chassis_secure": bool(status_reg & self.STATUS_CHASSIS_SECURE),
            "seals_intact": bool(status_reg & self.STATUS_SEALS_INTACT),
            "all_sensors_ok": bool(status_reg & self.STATUS_ALL_SENSORS_OK),
            "event_count": len(self.intrusion_events),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "IDS_STATUS":
                value = self._read_status_register()
            elif register == "EVENT_COUNT":
                value = len(self.intrusion_events)
            elif register == "LAST_EVENT":
                value = self.intrusion_events[-1]["type"] if self.intrusion_events else 0
            elif register == "SENSOR_STATUS":
                value = sum(1 << i for i, ok in enumerate(self.sensor_states.values()) if ok)
            elif register == "ALERT_STATUS":
                value = 1 if not all(self.sensor_states.values()) else 0
            elif register == "CHASSIS_STATUS":
                value = 1 if self.sensor_states.get("chassis", False) else 0
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

    # IDS-specific operations

    def get_sensor_states(self) -> OperationResult:
        """Get all sensor states"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        sensors = []
        for sensor_name, secure in self.sensor_states.items():
            sensors.append({
                "sensor": sensor_name,
                "secure": secure,
                "status": "OK" if secure else "ALERT",
            })

        self._record_operation(True)
        return OperationResult(True, data={
            "sensors": sensors,
            "all_secure": all(self.sensor_states.values()),
        })

    def get_intrusion_events(self, limit: int = 10) -> OperationResult:
        """Get recent intrusion events"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        events = self.intrusion_events[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "events": events,
            "total_events": len(self.intrusion_events),
            "returned": len(events),
        })

    def _read_status_register(self) -> int:
        """Read IDS status register"""
        status = 0

        if self.armed:
            status |= self.STATUS_ARMED

        if not all(self.sensor_states.values()):
            status |= self.STATUS_INTRUSION_DETECTED

        if self.sensor_states.get("chassis", False):
            status |= self.STATUS_CHASSIS_SECURE

        if self.sensor_states.get("seal_1", False) and self.sensor_states.get("seal_2", False):
            status |= self.STATUS_SEALS_INTACT

        if all(self.sensor_states.values()):
            status |= self.STATUS_ALL_SENSORS_OK

        return status
