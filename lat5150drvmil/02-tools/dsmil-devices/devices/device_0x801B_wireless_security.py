#!/usr/bin/env python3
"""
Device 0x801B: Wireless Security Controller

Manages wireless communication security including WiFi, Bluetooth, and
RF transmission control for TEMPEST and EMSEC compliance.

Device ID: 0x801B
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


class WirelessInterface(object):
    """Wireless interface types"""
    WIFI_2_4GHZ = 0
    WIFI_5GHZ = 1
    BLUETOOTH = 2
    NFC = 3
    GPS = 4


class WirelessSecurityDevice(DSMILDeviceBase):
    """Wireless Security Controller (0x801B)"""

    # Register map
    REG_WIRELESS_STATUS = 0x00
    REG_ENABLED_INTERFACES = 0x04
    REG_ENCRYPTION_STATUS = 0x08
    REG_RF_EMISSIONS = 0x0C

    # Status bits
    STATUS_RADIO_ENABLED = 0x01
    STATUS_ENCRYPTED = 0x02
    STATUS_TEMPEST_COMPLIANT = 0x04

    def __init__(self, device_id: int = 0x801B,
                 name: str = "Wireless Security",
                 description: str = "Wireless Communication Security Control"):
        super().__init__(device_id, name, description)

        # Device-specific state
        self.interfaces = {
            "WiFi_2.4GHz": {"type": WirelessInterface.WIFI_2_4GHZ, "enabled": False, "encrypted": True},
            "WiFi_5GHz": {"type": WirelessInterface.WIFI_5GHZ, "enabled": False, "encrypted": True},
            "Bluetooth": {"type": WirelessInterface.BLUETOOTH, "enabled": False, "encrypted": True},
        }
        self.tempest_mode = True

        # Register map
        self.register_map = {
            "WIRELESS_STATUS": {
                "offset": self.REG_WIRELESS_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Wireless security status"
            },
            "ENABLED_INTERFACES": {
                "offset": self.REG_ENABLED_INTERFACES,
                "size": 4,
                "access": "RO",
                "description": "Enabled wireless interfaces"
            },
            "ENCRYPTION_STATUS": {
                "offset": self.REG_ENCRYPTION_STATUS,
                "size": 4,
                "access": "RO",
                "description": "Encryption status"
            },
            "RF_EMISSIONS": {
                "offset": self.REG_RF_EMISSIONS,
                "size": 4,
                "access": "RO",
                "description": "RF emission level"
            },
        }

    def initialize(self) -> OperationResult:
        """Initialize Wireless Security device"""
        try:
            self.state = DeviceState.INITIALIZING

            # Default: all wireless disabled for security
            for interface in self.interfaces.values():
                interface["enabled"] = False
                interface["encrypted"] = True

            self.tempest_mode = True

            self.state = DeviceState.READY
            self._record_operation(True)

            return OperationResult(True, data={
                "interfaces": len(self.interfaces),
                "tempest_mode": self.tempest_mode,
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
        enabled_interfaces = sum(1 for i in self.interfaces.values() if i["enabled"])
        radio_enabled = enabled_interfaces > 0
        all_encrypted = all(i["encrypted"] for i in self.interfaces.values() if i["enabled"])

        return {
            "radio_enabled": radio_enabled,
            "encrypted": all_encrypted,
            "tempest_compliant": self.tempest_mode,
            "enabled_interfaces": enabled_interfaces,
            "total_interfaces": len(self.interfaces),
            "state": self.state.value,
        }

    def read_register(self, register: str) -> OperationResult:
        """Read a device register"""
        if register not in self.register_map:
            return OperationResult(False, error=f"Unknown register: {register}")

        try:
            if register == "WIRELESS_STATUS":
                status = self.STATUS_TEMPEST_COMPLIANT if self.tempest_mode else 0
                if any(i["enabled"] for i in self.interfaces.values()):
                    status |= self.STATUS_RADIO_ENABLED
                if all(i["encrypted"] for i in self.interfaces.values() if i["enabled"]):
                    status |= self.STATUS_ENCRYPTED
                value = status
            elif register == "ENABLED_INTERFACES":
                value = sum(1 for i in self.interfaces.values() if i["enabled"])
            elif register == "ENCRYPTION_STATUS":
                value = sum(1 for i in self.interfaces.values() if i["encrypted"])
            elif register == "RF_EMISSIONS":
                value = 0 if self.tempest_mode else 1
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

    def get_interface_list(self) -> OperationResult:
        """Get list of wireless interfaces"""
        if self.state != DeviceState.READY:
            return OperationResult(False, error="Device not ready")

        interfaces = []
        for name, info in self.interfaces.items():
            interfaces.append({
                "name": name,
                "type": info["type"],
                "enabled": info["enabled"],
                "encrypted": info["encrypted"],
            })

        self._record_operation(True)
        return OperationResult(True, data={"interfaces": interfaces})
