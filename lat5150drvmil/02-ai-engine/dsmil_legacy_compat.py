#!/usr/bin/env python3
"""
DSMIL Legacy Compatibility Layer
================================

This module provides a compatibility shim for v1.x code to work with v2.0 APIs.

⚠️ DEPRECATED: This compatibility layer is provided for gradual migration only.
   New code should use the native v2.0 APIs directly.

Status: Maintained until 2026 Q2
Removal: 2026 Q3 (v3.0.0)

Usage:
    # At the top of legacy scripts, replace old imports with:
    from dsmil_legacy_compat import *

    # Rest of your v1.x code works unchanged

Migration Guide: See DEPRECATION_PLAN.md
"""

import warnings
import sys
from typing import List, Dict, Optional, Any

# Import v2.0 components
from dsmil_device_database_extended import (
    ALL_DEVICES_EXTENDED,
    SAFE_DEVICES_EXTENDED,
    QUARANTINED_DEVICES_EXTENDED,
    GROUP_0_DEVICES,
    GROUP_1_DEVICES,
    GROUP_2_DEVICES,
    GROUP_3_DEVICES,
    GROUP_4_DEVICES,
    GROUP_5_DEVICES,
    GROUP_6_DEVICES,
    get_device_extended,
    get_devices_by_status_extended,
    get_devices_by_group_extended,
    DSMILDevice,
    DeviceStatus
)

from dsmil_integration_adapter import DSMILIntegrationAdapter

# Compatibility warning
_WARNING_SHOWN = False

def _show_deprecation_warning():
    """Show deprecation warning once per session"""
    global _WARNING_SHOWN
    if not _WARNING_SHOWN:
        warnings.warn(
            "\n"
            "=" * 80 + "\n"
            "DEPRECATION WARNING: You are using the legacy compatibility layer.\n\n"
            "This code will stop working in v3.0.0 (2026 Q3).\n\n"
            "Migration required:\n"
            "  - Update imports to use v2.0 APIs directly\n"
            "  - Replace dsmil_device_database with dsmil_device_database_extended\n"
            "  - Replace discovery functions with DSMILIntegrationAdapter\n"
            "  - Use unified entry point: python3 dsmil.py\n\n"
            "See DEPRECATION_PLAN.md for complete migration guide.\n"
            "=" * 80,
            DeprecationWarning,
            stacklevel=3
        )
        _WARNING_SHOWN = True


# ==============================================================================
# Database Compatibility
# ==============================================================================

# Expose v2.0 database as v1.x names
ALL_DEVICES = ALL_DEVICES_EXTENDED
SAFE_DEVICES = SAFE_DEVICES_EXTENDED
QUARANTINED_DEVICES = QUARANTINED_DEVICES_EXTENDED

# Device groups (v1.x only had 0-6, now we have 0-8 but keep same names)
DEVICE_GROUP_0 = GROUP_0_DEVICES
DEVICE_GROUP_1 = GROUP_1_DEVICES
DEVICE_GROUP_2 = GROUP_2_DEVICES
DEVICE_GROUP_3 = GROUP_3_DEVICES
DEVICE_GROUP_4 = GROUP_4_DEVICES
DEVICE_GROUP_5 = GROUP_5_DEVICES
DEVICE_GROUP_6 = GROUP_6_DEVICES


def get_device(device_id: int) -> Optional[DSMILDevice]:
    """
    Legacy get_device function

    ⚠️ DEPRECATED: Use get_device_extended() instead

    Args:
        device_id: Device ID (0-103, was 0-83 in v1.x)

    Returns:
        DSMILDevice object or None
    """
    _show_deprecation_warning()
    return get_device_extended(device_id)


def get_devices_by_status(status: DeviceStatus) -> Dict[int, DSMILDevice]:
    """
    Legacy get_devices_by_status function

    ⚠️ DEPRECATED: Use get_devices_by_status_extended() instead
    """
    _show_deprecation_warning()
    return get_devices_by_status_extended(status)


def get_devices_by_group(group_id: int) -> Dict[int, DSMILDevice]:
    """
    Legacy get_devices_by_group function

    ⚠️ DEPRECATED: Use get_devices_by_group_extended() instead
    """
    _show_deprecation_warning()
    return get_devices_by_group_extended(group_id)


# ==============================================================================
# Discovery Compatibility
# ==============================================================================

def discover_devices() -> List[int]:
    """
    Legacy discovery function

    ⚠️ DEPRECATED: Use DSMILIntegrationAdapter.discover_all_devices_cascading()

    Returns:
        List of token IDs (0x8000+) for backwards compatibility

    Migration:
        # Old
        tokens = discover_devices()

        # New
        adapter = DSMILIntegrationAdapter()
        device_ids = adapter.discover_all_devices_cascading()
    """
    _show_deprecation_warning()

    try:
        adapter = DSMILIntegrationAdapter()
        device_ids = adapter.discover_all_devices_cascading()

        # Convert device IDs (0-103) to token IDs (0x8000+) for compatibility
        # Each device has 3 tokens, we return the status token
        token_ids = [0x8000 + (device_id * 3) for device_id in device_ids]
        return token_ids

    except Exception as e:
        warnings.warn(f"Discovery failed: {e}", RuntimeWarning)
        return []


def auto_register_all_devices() -> tuple:
    """
    Legacy auto-registration function

    ⚠️ DEPRECATED: Use DSMILIntegrationAdapter.discover_all_devices_cascading()

    Returns:
        Tuple of (registry_dict, stats_dict) for backwards compatibility
    """
    _show_deprecation_warning()

    try:
        adapter = DSMILIntegrationAdapter()
        device_ids = adapter.discover_all_devices_cascading()

        # Build legacy registry format
        registry = {}
        for device_id in device_ids:
            device = get_device_extended(device_id)
            if device:
                token_id = 0x8000 + (device_id * 3)
                registry[token_id] = {
                    'device_id': device_id,
                    'name': device.name,
                    'status': device.status.value,
                    'token_id': token_id
                }

        # Build legacy stats format
        stats = {
            'total_discovered': len(device_ids),
            'safe': len([d for d in device_ids if get_device_extended(d).status == DeviceStatus.SAFE]),
            'unknown': len([d for d in device_ids if get_device_extended(d).status == DeviceStatus.UNKNOWN]),
            'quarantined': len([d for d in device_ids if get_device_extended(d).status == DeviceStatus.QUARANTINED])
        }

        return registry, stats

    except Exception as e:
        warnings.warn(f"Auto-registration failed: {e}", RuntimeWarning)
        return {}, {'total_discovered': 0, 'safe': 0, 'unknown': 0, 'quarantined': 0}


# ==============================================================================
# Activation Compatibility
# ==============================================================================

class ActivationResult:
    """Legacy activation result class for compatibility"""
    def __init__(self, success: bool, message: str):
        self.success = success
        self.message = message


class DSMILDeviceActivator:
    """
    Legacy device activator class

    ⚠️ DEPRECATED: Use DSMILIntegrationAdapter instead

    Migration:
        # Old
        activator = DSMILDeviceActivator()
        result = activator.activate_device(device_id)

        # New
        adapter = DSMILIntegrationAdapter()
        success = adapter.activate_device(device_id)
    """

    def __init__(self):
        _show_deprecation_warning()
        self.adapter = DSMILIntegrationAdapter()

    def activate_device(self, device_id: int) -> ActivationResult:
        """
        Activate a device (legacy interface)

        Args:
            device_id: Device ID (0-103)

        Returns:
            ActivationResult object with .success and .message attributes
        """
        try:
            success = self.adapter.activate_device(device_id)
            if success:
                return ActivationResult(True, f"Device {device_id} activated successfully")
            else:
                return ActivationResult(False, f"Device {device_id} activation failed")
        except Exception as e:
            return ActivationResult(False, f"Activation error: {e}")

    def deactivate_device(self, device_id: int) -> ActivationResult:
        """
        Deactivate a device (legacy interface)

        Args:
            device_id: Device ID (0-103)

        Returns:
            ActivationResult object
        """
        try:
            success = self.adapter.deactivate_device(device_id)
            if success:
                return ActivationResult(True, f"Device {device_id} deactivated successfully")
            else:
                return ActivationResult(False, f"Device {device_id} deactivation failed")
        except Exception as e:
            return ActivationResult(False, f"Deactivation error: {e}")


# ==============================================================================
# Utility Functions
# ==============================================================================

def get_migration_info() -> Dict[str, Any]:
    """
    Get information about migration status and required changes

    Returns:
        Dictionary with migration guidance
    """
    return {
        'compatibility_layer_version': '2.0.0',
        'removal_date': '2026-Q3',
        'deprecation_status': 'active',
        'migration_required': True,
        'migration_guide': 'DEPRECATION_PLAN.md',
        'new_entry_point': 'dsmil.py',
        'breaking_changes': [
            'Discovery returns device IDs (0-103) instead of token IDs (0x8000+)',
            'Activation returns boolean instead of ActivationResult object',
            'Database extended from 84 to 104 devices',
            'New driver: dsmil-104dev.ko replaces dsmil-84dev.ko'
        ],
        'compatibility_notes': [
            'This layer converts between v1.x and v2.0 APIs',
            'Performance may be slightly degraded',
            'Not all v1.x features are supported',
            'Critical bugs only will be fixed'
        ]
    }


def check_migration_needed() -> bool:
    """
    Check if code needs migration

    Returns:
        True if using compatibility layer (migration recommended)
    """
    return True  # Always true when using this module


# ==============================================================================
# Module Initialization
# ==============================================================================

# Show warning on import
_show_deprecation_warning()

# Provide helpful message when run directly
if __name__ == "__main__":
    print("=" * 80)
    print("DSMIL Legacy Compatibility Layer")
    print("=" * 80)
    print()
    print("⚠️  This module provides backwards compatibility for v1.x code.")
    print("⚠️  It is DEPRECATED and will be removed in v3.0.0 (2026 Q3).")
    print()
    print("Migration Information:")
    print("-" * 80)

    info = get_migration_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"\n{key}:")
            for item in value:
                print(f"  - {item}")
        else:
            print(f"{key}: {value}")

    print()
    print("=" * 80)
    print("Next Steps:")
    print("  1. Read DEPRECATION_PLAN.md for migration guide")
    print("  2. Update imports to use v2.0 APIs directly")
    print("  3. Test with: python3 dsmil.py diagnostics")
    print("  4. Use unified entry point: python3 dsmil.py")
    print("=" * 80)
