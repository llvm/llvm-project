#!/usr/bin/env python3
"""
Device 0x802B: Hardware Packet Filter

Hardware-accelerated network packet filtering and inspection for
deep packet inspection (DPI), firewall, and intrusion prevention.

Device ID: 0x802B
Group: 2 (Network/Communications)
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


class FilterAction(object):
    """Packet filter actions"""
    ALLOW = 0
    DENY = 1
    LOG = 2
    INSPECT = 3


class FilterRule(object):
    """Filter rule structure"""
    def __init__(self, rule_id: int, protocol: str, port: int, action: int):
        self.rule_id = rule_id
        self.protocol = protocol
        self.port = port
        self.action = action
        self.match_count = 0


class PacketFilterDevice(DSMILDeviceBase):
    """Hardware Packet Filter (0x802B)"""

    # Register map
    REG_FILTER_STATUS = 0x00
    REG_RULE_COUNT = 0x04
    REG_PACKETS_FILTERED = 0x08
    REG_PACKETS_BLOCKED = 0x0C

    # Status bits
    STATUS_FILTER_ACTIVE = 0x01
    STATUS_DPI_ENABLED = 0x02
    STATUS_IPS_MODE = 0x04

    def __init__(self, device_id: int = 0x802B,
                 name: str = "Packet Filter",
                 description: str = "Hardware Packet Filtering and Inspection"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.filter_enabled = True
        self.dpi_enabled = True
        self.ips_mode = False
        self.rules = {
            1: FilterRule(1, "TCP", 22, FilterAction.ALLOW),
            2: FilterRule(2, "TCP", 443, FilterAction.ALLOW),
            3: FilterRule(3, "TCP", 23, FilterAction.DENY),  # Telnet blocked
        }
        self.packets_filtered = 0
        self.packets_blocked = 0

        # Register map
        self.register_map = {
            "FILTER_STATUS": {
                "offset": self.REG_FILTER_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Packet filter status"
            },
            "RULE_COUNT": {
                "offset": self.REG_RULE_COUNT,
                "size": 4,
                "access": "RO",
                "description": "Number of active rules"
            },
            "PACKETS_FILTERED": {
                "offset": self.REG_PACKETS_FILTERED,
                "size": 4,
                "access": "RO",
                "description": "Total packets filtered"
            },
            "PACKETS_BLOCKED": {
                "offset": self.REG_PACKETS_BLOCKED,
                "size": 4,
                "access": "RO",
                "description": "Total packets blocked"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Packet Filter device"""
        try:
            self.state = DeviceState.INITIALIZING

            self.filter_enabled = True
            self.dpi_enabled = True
            self.ips_mode = False
            self.packets_filtered = 0
            self.packets_blocked = 0

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "filter_enabled": self.filter_enabled,
                "dpi_enabled": self.dpi_enabled,
                "rules": len(self.rules),
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
            DeviceCapability.DMA_CAPABLE,
        ]

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        return {
            "filter_active": self.filter_enabled,
            "dpi_enabled": self.dpi_enabled,
            "ips_mode": self.ips_mode,
            "active_rules": len(self.rules),
            "packets_filtered": self.packets_filtered,
            "packets_blocked": self.packets_blocked,
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "FILTER_STATUS":
                status = 0
                if self.filter_enabled:
                    status |= self.STATUS_FILTER_ACTIVE
                if self.dpi_enabled:
                    status |= self.STATUS_DPI_ENABLED
                if self.ips_mode:
                    status |= self.STATUS_IPS_MODE
                value = status
            elif register == "RULE_COUNT":
                value = len(self.rules)
            elif register == "PACKETS_FILTERED":
                value = self.packets_filtered
            elif register == "PACKETS_BLOCKED":
                value = self.packets_blocked
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

    def get_filter_rules(self) -> OperationResult:
        """Get all filter rules"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        rules = []
        for rule in self.rules.values():
            rules.append({
                "rule_id": rule.rule_id,
                "protocol": rule.protocol,
                "port": rule.port,
                "action": rule.action,
                "match_count": rule.match_count,
            })

        self._record_operation(True)
        return OperationResult(True, data={"rules": rules})

    def get_statistics(self) -> Dict[str, Any]:
        """Get packet filter statistics"""
        stats = super().get_statistics()
        stats.update({
            "packets_filtered": self.packets_filtered,
            "packets_blocked": self.packets_blocked,
            "active_rules": len(self.rules),
        })
        return stats
