#!/usr/bin/env python3
"""
Device 0x8004: Event Logger

System event logging and monitoring for all platform operations.
Captures system, application, security, and hardware events.

Device ID: 0x8004
Group: 0 (Core Security)
Risk Level: SAFE (Read-only event access)

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


class EventLevel:
    """Event severity levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class EventType:
    """Event type categories"""
    SYSTEM = 0
    APPLICATION = 1
    SECURITY = 2
    HARDWARE = 3
    NETWORK = 4
    STORAGE = 5
    DRIVER = 6


class EventLoggerDevice(DSMILDeviceBase):
    """Event Logger (0x8004)"""

    # Register map
    REG_LOGGER_STATUS = 0x00
    REG_EVENT_COUNT = 0x04
    REG_EVENT_CAPACITY = 0x08
    REG_EVENT_POSITION = 0x0C
    REG_LAST_LEVEL = 0x10
    REG_ERROR_COUNT = 0x14
    REG_CRITICAL_COUNT = 0x18

    # Status bits
    STATUS_LOGGER_ACTIVE = 0x01
    STATUS_LOGGER_FULL = 0x02
    STATUS_ERRORS_PRESENT = 0x04
    STATUS_CRITICAL_EVENTS = 0x08
    STATUS_REAL_TIME = 0x10
    STATUS_REMOTE_LOGGING = 0x20

    def __init__(self, device_id: int = 0x8004,
                 name: str = "Event Logger",
                 description: str = "System Event Logging and Monitoring"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.events = []
        self.max_events = 50000
        self.real_time_logging = True
        self.remote_logging = False
        self.error_count = 0
        self.critical_count = 0

        # Initialize with sample events
        self._initialize_sample_events()

        # Register map
        self.register_map = {
            "LOGGER_STATUS": {
                "offset": self.REG_LOGGER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Event logger status"
            },
            "EVENT_COUNT": {
                "offset": self.REG_EVENT_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of logged events"
            },
            "EVENT_CAPACITY": {
                "offset": self.REG_EVENT_CAPACITY,
                "size": 4,
                "access": "RO",
                "description": "Maximum event capacity"
            },
            "EVENT_POSITION": {
                "offset": self.REG_EVENT_POSITION,
                "size": 4,
                "access": "RO",
                "description": "Current event write position"
            },
            "LAST_LEVEL": {
                "offset": self.REG_LAST_LEVEL,
                "size": 4,
                "access": "RO",
                "description": "Last event severity level"
            },
            "ERROR_COUNT": {
                "offset": self.REG_ERROR_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total error events"
            },
            "CRITICAL_COUNT": {
                "offset": self.REG_CRITICAL_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Total critical events"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Event Logger device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Initialize event log
            self._initialize_sample_events()
            self.real_time_logging = True
            self.remote_logging = False
            self._update_counters()

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "total_events": len(self.events),
                "capacity": self.max_events,
                "real_time": self.real_time_logging,
                "errors": self.error_count,
                "critical": self.critical_count,
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
            "logger_active": bool(status_reg & self.STATUS_LOGGER_ACTIVE),
            "logger_full": bool(status_reg & self.STATUS_LOGGER_FULL),
            "errors_present": bool(status_reg & self.STATUS_ERRORS_PRESENT),
            "critical_events": bool(status_reg & self.STATUS_CRITICAL_EVENTS),
            "real_time": bool(status_reg & self.STATUS_REAL_TIME),
            "remote_logging": bool(status_reg & self.STATUS_REMOTE_LOGGING),
            "event_count": len(self.events),
            "capacity": self.max_events,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "LOGGER_STATUS":
                value = self._read_status_register()
            elif register == "EVENT_COUNT":
                value = len(self.events)
            elif register == "EVENT_CAPACITY":
                value = self.max_events
            elif register == "EVENT_POSITION":
                value = len(self.events)
            elif register == "LAST_LEVEL":
                value = self.events[-1]['level'] if self.events else 0
            elif register == "ERROR_COUNT":
                value = self.error_count
            elif register == "CRITICAL_COUNT":
                value = self.critical_count
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

    # Event Logger specific operations

    def get_recent_events(self, limit: int = 100) -> OperationResult:
        """Get recent events"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        events = self.events[-limit:]

        self._record_operation(True)
        return OperationResult(True, data={
            "events": events,
            "total": len(events),
            "showing": f"Last {min(limit, len(self.events))} of {len(self.events)}",
        })

    def get_events_by_level(self, level: int) -> OperationResult:
        """Get events by severity level"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        events = [e for e in self.events if e['level'] == level]

        self._record_operation(True)
        return OperationResult(True, data={
            "events": events,
            "total": len(events),
            "level": self._get_level_name(level),
        })

    def get_events_by_type(self, event_type: int) -> OperationResult:
        """Get events by type"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        events = [e for e in self.events if e['type'] == event_type]

        self._record_operation(True)
        return OperationResult(True, data={
            "events": events,
            "total": len(events),
            "type": self._get_type_name(event_type),
        })

    def get_error_events(self) -> OperationResult:
        """Get all error and critical events"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        events = [e for e in self.events
                 if e['level'] >= EventLevel.ERROR]

        self._record_operation(True)
        return OperationResult(True, data={
            "events": events,
            "total": len(events),
            "errors": self.error_count,
            "critical": self.critical_count,
        })

    def get_summary(self) -> OperationResult:
        """Get event log summary"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        # Count by level
        by_level = {}
        for event in self.events:
            level = event['level']
            by_level[level] = by_level.get(level, 0) + 1

        # Count by type
        by_type = {}
        for event in self.events:
            etype = event['type']
            by_type[etype] = by_type.get(etype, 0) + 1

        summary = {
            "total_events": len(self.events),
            "capacity": self.max_events,
            "utilization": f"{len(self.events)/self.max_events*100:.1f}%",
            "by_level": {
                self._get_level_name(level): count
                for level, count in by_level.items()
            },
            "by_type": {
                self._get_type_name(etype): count
                for etype, count in by_type.items()
            },
            "error_count": self.error_count,
            "critical_count": self.critical_count,
            "real_time": self.real_time_logging,
        }

        self._record_operation(True)
        return OperationResult(True, data=summary)

    def get_statistics(self) -> Dict[str, Any]:
        """Get event logger statistics"""
        stats = super().get_statistics()

        stats.update({
            "total_events": len(self.events),
            "capacity": self.max_events,
            "utilization_percent": len(self.events)/self.max_events*100,
            "error_count": self.error_count,
            "critical_count": self.critical_count,
            "real_time_logging": self.real_time_logging,
        })

        return stats

    # Internal helper methods

    def _read_status_register(self) -> int:
        """Read event logger status register (simulated)"""
        status = self.STATUS_LOGGER_ACTIVE

        if len(self.events) >= self.max_events:
            status |= self.STATUS_LOGGER_FULL

        if self.error_count > 0:
            status |= self.STATUS_ERRORS_PRESENT

        if self.critical_count > 0:
            status |= self.STATUS_CRITICAL_EVENTS

        if self.real_time_logging:
            status |= self.STATUS_REAL_TIME

        if self.remote_logging:
            status |= self.STATUS_REMOTE_LOGGING

        return status

    def _initialize_sample_events(self):
        """Initialize with sample events"""
        current_time = time.time()

        self.events = [
            {
                "timestamp": current_time - 7200,
                "level": EventLevel.INFO,
                "type": EventType.SYSTEM,
                "message": "System startup completed",
                "source": "kernel",
                "code": 1000,
            },
            {
                "timestamp": current_time - 6000,
                "level": EventLevel.INFO,
                "type": EventType.DRIVER,
                "message": "DSMIL driver loaded successfully",
                "source": "dsmil_driver",
                "code": 2001,
            },
            {
                "timestamp": current_time - 5400,
                "level": EventLevel.WARNING,
                "type": EventType.HARDWARE,
                "message": "Temperature threshold warning",
                "source": "thermal_monitor",
                "code": 3010,
            },
            {
                "timestamp": current_time - 4800,
                "level": EventLevel.INFO,
                "type": EventType.SECURITY,
                "message": "TPM initialized successfully",
                "source": "tpm_manager",
                "code": 4005,
            },
            {
                "timestamp": current_time - 3600,
                "level": EventLevel.ERROR,
                "type": EventType.NETWORK,
                "message": "Network interface timeout",
                "source": "network_manager",
                "code": 5100,
            },
            {
                "timestamp": current_time - 2400,
                "level": EventLevel.INFO,
                "type": EventType.APPLICATION,
                "message": "Application service started",
                "source": "app_manager",
                "code": 6001,
            },
            {
                "timestamp": current_time - 1200,
                "level": EventLevel.WARNING,
                "type": EventType.STORAGE,
                "message": "Disk space low on volume /data",
                "source": "storage_monitor",
                "code": 7050,
            },
            {
                "timestamp": current_time - 600,
                "level": EventLevel.INFO,
                "type": EventType.SYSTEM,
                "message": "Configuration updated",
                "source": "config_manager",
                "code": 1020,
            },
        ]

    def _update_counters(self):
        """Update error and critical event counters"""
        self.error_count = len([e for e in self.events
                               if e['level'] == EventLevel.ERROR])
        self.critical_count = len([e for e in self.events
                                  if e['level'] == EventLevel.CRITICAL])

    def _get_level_name(self, level: int) -> str:
        """Get event level name"""
        names = {
            EventLevel.DEBUG: "Debug",
            EventLevel.INFO: "Info",
            EventLevel.WARNING: "Warning",
            EventLevel.ERROR: "Error",
            EventLevel.CRITICAL: "Critical",
        }
        return names.get(level, "Unknown")

    def _get_type_name(self, event_type: int) -> str:
        """Get event type name"""
        names = {
            EventType.SYSTEM: "System",
            EventType.APPLICATION: "Application",
            EventType.SECURITY: "Security",
            EventType.HARDWARE: "Hardware",
            EventType.NETWORK: "Network",
            EventType.STORAGE: "Storage",
            EventType.DRIVER: "Driver",
        }
        return names.get(event_type, "Unknown")


def main():
    """Test Event Logger device"""
    print("=" * 80)
    print("Device 0x8004: Event Logger - Test")
    print("=" * 80)

    device = EventLoggerDevice()

    # Initialize
    print("\n1. Initializing device...")
    result = device.initialize()
    print(f"   Success: {result.success}")
    if result.success:
        print(f"   Events: {result.data['total_events']}")
        print(f"   Errors: {result.data['errors']}")
        print(f"   Critical: {result.data['critical']}")

    # Get status
    print("\n2. Getting device status...")
    status = device.get_status()
    print(f"   Logger active: {status['logger_active']}")
    print(f"   Errors present: {status['errors_present']}")
    print(f"   Event count: {status['event_count']}")

    # Get recent events
    print("\n3. Getting recent events...")
    result = device.get_recent_events(limit=5)
    if result.success:
        print(f"   Retrieved: {result.data['total']} events")
        for event in result.data['events'][-3:]:
            print(f"   - [{device._get_level_name(event['level'])}] {event['message']}")

    # Get error events
    print("\n4. Getting error events...")
    result = device.get_error_events()
    if result.success:
        print(f"   Total errors: {result.data['total']}")
        print(f"   Error count: {result.data['errors']}")
        print(f"   Critical count: {result.data['critical']}")

    # Get summary
    print("\n5. Getting summary...")
    result = device.get_summary()
    if result.success:
        print(f"   Total events: {result.data['total_events']}")
        print(f"   Utilization: {result.data['utilization']}")
        print(f"   By level: {result.data['by_level']}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
