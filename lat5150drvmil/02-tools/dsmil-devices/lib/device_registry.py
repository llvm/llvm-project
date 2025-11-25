#!/usr/bin/env python3
"""
DSMIL Device Registry

Central registry for all DSMIL device integrations. Manages device
discovery, initialization, and access control.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

from typing import Dict, List, Optional, Type
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from device_base import DSMILDeviceBase, DeviceState, OperationResult


class DeviceGroup(Enum):
    """DSMIL device groups"""
    GROUP_0_CORE_SECURITY = 0
    GROUP_1_EXTENDED_SECURITY = 1
    GROUP_2_NETWORK_COMMS = 2
    GROUP_3_DATA_PROCESSING = 3
    GROUP_4_STORAGE_MANAGEMENT = 4
    GROUP_5_PERIPHERAL_CONTROL = 5
    GROUP_6_TRAINING_SIMULATION = 6


class DeviceRiskLevel(Enum):
    """Device risk classification"""
    SAFE = "safe"
    MONITORED = "monitored"
    CAUTION = "caution"
    RISKY = "risky"
    QUARANTINED = "quarantined"


class DeviceRegistration:
    """Device registration information"""

    def __init__(self, device_class: Type[DSMILDeviceBase],
                 risk_level: DeviceRiskLevel,
                 group: DeviceGroup,
                 enabled: bool = True):
        self.device_class = device_class
        self.risk_level = risk_level
        self.group = group
        self.enabled = enabled
        self.instance = None


class DSMILDeviceRegistry:
    """Central registry for DSMIL devices"""

    def __init__(self):
        self._devices: Dict[int, DeviceRegistration] = {}
        self._initialized_devices: Dict[int, DSMILDeviceBase] = {}

    def register_device(self, device_id: int, device_class: Type[DSMILDeviceBase],
                       risk_level: DeviceRiskLevel, group: DeviceGroup,
                       enabled: bool = True):
        """
        Register a device implementation

        Args:
            device_id: Device ID (0x8000-0x806B)
            device_class: Device implementation class
            risk_level: Risk classification
            group: Device group
            enabled: Whether device is enabled
        """
        if device_id in self._devices:
            raise ValueError(f"Device 0x{device_id:04X} already registered")

        if device_id < 0x8000 or device_id > 0x806B:
            raise ValueError(f"Invalid device ID 0x{device_id:04X}")

        self._devices[device_id] = DeviceRegistration(
            device_class, risk_level, group, enabled
        )

    def unregister_device(self, device_id: int):
        """Unregister a device"""
        if device_id in self._devices:
            # Cleanup if initialized
            if device_id in self._initialized_devices:
                del self._initialized_devices[device_id]
            del self._devices[device_id]

    def get_device(self, device_id: int) -> Optional[DSMILDeviceBase]:
        """
        Get device instance (initializes if needed)

        Args:
            device_id: Device ID

        Returns:
            Device instance or None if not registered/enabled
        """
        if device_id not in self._devices:
            return None

        registration = self._devices[device_id]
        if not registration.enabled:
            return None

        # Initialize if needed
        if device_id not in self._initialized_devices:
            instance = registration.device_class(
                device_id,
                registration.device_class.__name__,
                ""
            )
            self._initialized_devices[device_id] = instance

        return self._initialized_devices[device_id]

    def get_devices_by_group(self, group: DeviceGroup) -> List[DSMILDeviceBase]:
        """Get all devices in a group"""
        devices = []
        for device_id, registration in self._devices.items():
            if registration.group == group and registration.enabled:
                device = self.get_device(device_id)
                if device:
                    devices.append(device)
        return devices

    def get_devices_by_risk(self, risk_level: DeviceRiskLevel) -> List[DSMILDeviceBase]:
        """Get all devices with specific risk level"""
        devices = []
        for device_id, registration in self._devices.items():
            if registration.risk_level == risk_level and registration.enabled:
                device = self.get_device(device_id)
                if device:
                    devices.append(device)
        return devices

    def get_all_devices(self) -> Dict[int, DSMILDeviceBase]:
        """Get all registered and enabled devices"""
        devices = {}
        for device_id in self._devices.keys():
            device = self.get_device(device_id)
            if device:
                devices[device_id] = device
        return devices

    def initialize_all(self) -> Dict[int, OperationResult]:
        """
        Initialize all registered devices

        Returns:
            Dictionary mapping device IDs to initialization results
        """
        results = {}
        for device_id in self._devices.keys():
            device = self.get_device(device_id)
            if device:
                results[device_id] = device.initialize()
        return results

    def get_registry_summary(self) -> Dict:
        """Get summary of registered devices"""
        total = len(self._devices)
        enabled = sum(1 for r in self._devices.values() if r.enabled)
        initialized = len(self._initialized_devices)

        by_group = {}
        by_risk = {}

        for registration in self._devices.values():
            group_name = registration.group.name
            risk_name = registration.risk_level.value

            by_group[group_name] = by_group.get(group_name, 0) + 1
            by_risk[risk_name] = by_risk.get(risk_name, 0) + 1

        return {
            "total_registered": total,
            "enabled": enabled,
            "initialized": initialized,
            "by_group": by_group,
            "by_risk": by_risk,
        }

    def list_devices(self) -> List[Dict]:
        """List all registered devices with details"""
        devices = []
        for device_id, registration in sorted(self._devices.items()):
            device_info = {
                "device_id": f"0x{device_id:04X}",
                "name": registration.device_class.__name__,
                "group": registration.group.name,
                "risk_level": registration.risk_level.value,
                "enabled": registration.enabled,
                "initialized": device_id in self._initialized_devices,
            }

            if device_id in self._initialized_devices:
                instance = self._initialized_devices[device_id]
                device_info["state"] = instance.state.value

            devices.append(device_info)

        return devices


# Global registry instance
_registry = DSMILDeviceRegistry()


def get_registry() -> DSMILDeviceRegistry:
    """Get the global device registry"""
    return _registry
