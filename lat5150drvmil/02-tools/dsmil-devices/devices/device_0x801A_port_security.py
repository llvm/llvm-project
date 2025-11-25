#!/usr/bin/env python3
"""
Device 0x801A: Port Security Controller

Hardware-level physical port security and access control. Manages USB,
Ethernet, serial, and other physical port security policies.

Device ID: 0x801A
Group: 1 (Extended Security)
Risk Level: MONITORED (needs testing)

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

from device_base import (
    DSMILDeviceBase, DeviceCapability, DeviceState, OperationResult
)
from typing import Dict, List, Optional, Any


class PortType(object):
    """Physical port types"""
    USB_A = 0
    USB_C = 1
    ETHERNET = 2
    SERIAL = 3
    HDMI = 4
    AUDIO = 5


class PortPolicy(object):
    """Port security policies"""
    ENABLED = 0x01
    WHITELIST_ONLY = 0x02
    READ_ONLY = 0x04
    LOGGED = 0x08


class PortSecurityDevice(DSMILDeviceBase):
    """Port Security Controller (0x801A)"""

    # Register map
    REG_PORT_STATUS = 0x00
    REG_POLICY = 0x04
    REG_ACTIVE_PORTS = 0x08
    REG_BLOCKED_COUNT = 0x0C

    def __init__(self, device_id: int = 0x801A,
                 name: str = "Port Security",
                 description: str = "Physical Port Security Control"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.ports = {
            "USB_A_1": {"type": PortType.USB_A, "enabled": True, "policy": PortPolicy.LOGGED},
            "USB_A_2": {"type": PortType.USB_A, "enabled": True, "policy": PortPolicy.LOGGED},
            "USB_C_1": {"type": PortType.USB_C, "enabled": True, "policy": PortPolicy.LOGGED},
            "ETHERNET_1": {"type": PortType.ETHERNET, "enabled": True, "policy": PortPolicy.ENABLED},
        }
        self.blocked_attempts = 0

        # Register map
        self.register_map = {
            "PORT_STATUS": {
                "offset": self.REG_PORT_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Port security status"
            },
            "POLICY": {
                "offset": self.REG_POLICY,
                "size": 4,
                "access": "RO",
                "description": "Active security policy"
            },
            "ACTIVE_PORTS": {
                "offset": self.REG_ACTIVE_PORTS,
                "size": 4,
                "access": "RO",
                "description": "Number of active ports"
            },
            "BLOCKED_COUNT": {
                "offset": self.REG_BLOCKED_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Blocked access attempts"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Port Security device"""
        try:
            self.state = DeviceState.INITIALIZING
            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "ports": len(self.ports),
                "enabled": sum(1 for p in self.ports.values() if p["enabled"]),
            })

        except Exception as e:
            self.state = DeviceState.ERROR
            self._record_operation(False, str(e))
            return OperationResult(False, error=str(e))

    def get_capabilities(self) -> List[DeviceCapability]:
        """Get device capabilities"""
        return [
            DeviceCapability.READ_WRITE,
            DeviceCapability.CONFIGURATION,
            DeviceCapability.STATUS_REPORTING,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        enabled_ports = sum(1 for p in self.ports.values() if p["enabled"])

        return {
            "total_ports": len(self.ports),
            "enabled_ports": enabled_ports,
            "blocked_attempts": self.blocked_attempts,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "PORT_STATUS":
                value = 0x01  # Ready
            elif register == "POLICY":
                value = PortPolicy.LOGGED
            elif register == "ACTIVE_PORTS":
                value = sum(1 for p in self.ports.values() if p["enabled"])
            elif register == "BLOCKED_COUNT":
                value = self.blocked_attempts
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

    def get_port_list(self) -> OperationResult:
        """Get list of all ports with status"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        port_list = []
        for name, info in self.ports.items():
            port_list.append({
                "name": name,
                "type": info["type"],
                "enabled": info["enabled"],
                "policy": info["policy"],
            })

        self._record_operation(True)
        return OperationResult(True, data={"ports": port_list})
