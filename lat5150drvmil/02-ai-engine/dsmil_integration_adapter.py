#!/usr/bin/env python3
"""
DSMIL Integration Adapter
==========================
Unified adapter connecting legacy DSMIL Python tools with the new
104-device kernel driver v5.2.0.

This adapter provides:
- Backward compatibility for existing Python control centres
- Automatic translation between old and new interfaces
- Cascading device discovery for all 104 devices
- Integration with ML-enhanced activation systems
- Unified API for all DSMIL operations

Author: LAT5150DRVMIL Integration Team
Version: 1.0.0
Driver Compatibility: dsmil-104dev v5.2.0
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Add paths for existing modules
sys.path.insert(0, str(Path(__file__).parent))

# Import new driver interface
from dsmil_driver_interface import (
    DSMILDriverInterface,
    SystemStatus,
    DeviceInfo,
    BiosStatus,
    BiosID,
    check_driver_loaded,
    discover_all_devices,
)

# Import extended device database
from dsmil_device_database_extended import (
    ALL_DEVICES_EXTENDED,
    get_device_extended,
    get_device_by_token,
    get_token_range,
    get_statistics_extended,
    QUARANTINED_DEVICES_EXTENDED,
    SAFE_DEVICES_EXTENDED,
)

# Try to import legacy activation and discovery modules
try:
    from dsmil_integrated_activation import DSMILIntegratedActivation
    LEGACY_ACTIVATION_AVAILABLE = True
except ImportError:
    LEGACY_ACTIVATION_AVAILABLE = False
    logging.warning("Legacy activation module not available")

try:
    from dsmil_device_activation import DSMILDeviceActivator
    LEGACY_ACTIVATOR_AVAILABLE = True
except ImportError:
    LEGACY_ACTIVATOR_AVAILABLE = False
    logging.warning("Legacy device activator not available")

try:
    from dsmil_subsystem_controller import DSMILSubsystemController
    LEGACY_CONTROLLER_AVAILABLE = True
except ImportError:
    LEGACY_CONTROLLER_AVAILABLE = False
    logging.warning("Legacy subsystem controller not available")


logger = logging.getLogger(__name__)


# ============================================================================
# ACTIVATION STATUS
# ============================================================================

class ActivationStatus(Enum):
    """Device activation status"""
    NOT_ACTIVATED = "not_activated"
    ACTIVATING = "activating"
    ACTIVATED = "activated"
    FAILED = "failed"
    QUARANTINED = "quarantined"


@dataclass
class DeviceStatus:
    """Complete device status"""
    device_id: int
    name: str
    token_status: int
    token_config: int
    token_data: int
    is_online: bool
    is_ready: bool
    is_safe: bool
    is_quarantined: bool
    activation_status: ActivationStatus
    group: int


# ============================================================================
# INTEGRATION ADAPTER CLASS
# ============================================================================

class DSMILIntegrationAdapter:
    """
    Unified adapter for DSMIL control centre and discovery integration

    Provides single interface for:
    - Device discovery (cascading, ML-enhanced, token-based)
    - Device activation (IOCTL, sysfs, SMI methods)
    - System monitoring (thermal, BIOS, authentication)
    - Legacy tool compatibility
    """

    def __init__(self):
        """Initialize integration adapter"""
        self.driver = DSMILDriverInterface()
        self.device_cache: Dict[int, DeviceStatus] = {}
        self.activation_history: List[Tuple[int, bool, str]] = []

        # Check driver availability
        self.driver_available = check_driver_loaded()
        if not self.driver_available:
            logger.warning("DSMIL driver not loaded")

        # Detect backend mode (simulated vs real) if possible
        from dsmil_driver_interface import get_backend_name
        backend = get_backend_name()
        self.backend_name = backend
        self.simulated_backend = bool(backend and backend.startswith("simulated"))
        if self.simulated_backend:
            logger.warning("DSMIL running with simulated SMBIOS backend – no real firmware token access")

        # Initialize legacy components if available
        self.legacy_controller = None
        if LEGACY_CONTROLLER_AVAILABLE:
            try:
                self.legacy_controller = DSMILSubsystemController()
                logger.info("Legacy controller initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize legacy controller: {e}")

        logger.info("DSMIL Integration Adapter initialized")

    # ========================================================================
    # CASCADING DEVICE DISCOVERY
    # ========================================================================

    def discover_all_devices_cascading(self, progress_callback=None) -> List[int]:
        """
        Cascading device discovery - scans all 104 devices systematically

        Discovery cascades through:
        1. Driver IOCTL token read (primary method)
        2. Sysfs token enumeration (fallback)
        3. Database validation (quarantine check)
        4. Legacy tool integration (compatibility)

        Returns list of discovered device IDs
        """
        discovered = []

        if not self.driver_available:
            logger.error("Driver not available for discovery")
            return discovered

        if progress_callback:
            progress_callback("Starting cascading device discovery...")

        # Open driver
        if not self.driver.open():
            logger.error("Failed to open driver")
            return discovered

        try:
            # Phase 1: IOCTL-based discovery (primary)
            if progress_callback:
                progress_callback("Phase 1: IOCTL token scanning...")

            for device_id in range(104):  # 0-103
                if progress_callback and device_id % 10 == 0:
                    progress_callback(f"  Scanning device {device_id}/104...")

                # Check if device is quarantined
                device_token_base = 0x8000 + (device_id * 3)
                if device_token_base in QUARANTINED_DEVICES_EXTENDED:
                    logger.info(f"Skipping quarantined device {device_id} (0x{device_token_base:04X})")
                    continue

                # Try to read status token
                device_info = self.driver.get_device_info(device_id)
                if device_info:
                    discovered.append(device_id)
                    logger.debug(f"Discovered device {device_id} via IOCTL")

                    # Cache device status
                    self._cache_device_status(device_id, device_info)

            # Phase 2: Database validation
            if progress_callback:
                progress_callback(f"Phase 2: Validating {len(discovered)} discovered devices...")

            validated = []
            for device_id in discovered:
                device = get_device_extended(device_id)
                if device:
                    validated.append(device_id)
                    logger.debug(f"Validated device {device_id}: {device.name}")

            logger.info(f"Cascading discovery complete: {len(validated)}/104 devices discovered and validated")

            if progress_callback:
                progress_callback(f"Discovery complete: {len(validated)} devices")

            return validated

        finally:
            self.driver.close()

    def _cache_device_status(self, device_id: int, device_info: DeviceInfo):
        """Cache device status for quick lookup"""
        device = get_device_extended(device_id)

        if not device:
            return

        # Determine activation status
        is_online = (device_info.status & 0x01) != 0
        is_ready = (device_info.status & 0x02) != 0

        if device.device_id in QUARANTINED_DEVICES_EXTENDED:
            activation_status = ActivationStatus.QUARANTINED
        elif is_online and is_ready:
            activation_status = ActivationStatus.ACTIVATED
        else:
            activation_status = ActivationStatus.NOT_ACTIVATED

        # Create full status object
        status = DeviceStatus(
            device_id=device_id,
            name=device.name,
            token_status=device_info.status,
            token_config=device_info.config,
            token_data=device_info.data,
            is_online=is_online,
            is_ready=is_ready,
            is_safe=device.device_id in SAFE_DEVICES_EXTENDED,
            is_quarantined=device.device_id in QUARANTINED_DEVICES_EXTENDED,
            activation_status=activation_status,
            group=device.group.value,
        )

        self.device_cache[device_id] = status

    # ========================================================================
    # DEVICE ACTIVATION
    # ========================================================================

    def activate_device(self, device_id: int, force: bool = False) -> bool:
        """
        Activate a single device with safety checks

        Args:
            device_id: Device ID (0-103)
            force: Skip safety checks (dangerous!)

        Returns:
            True if activation successful
        """
        device = get_device_extended(device_id)
        if not device:
            logger.error(f"Unknown device {device_id}")
            return False

        # If backend is simulated, block activation unless explicitly forced
        if self.simulated_backend and not force:
            logger.warning(
                "Activation blocked in simulated backend (device %d: %s). "
                "Use force=True only for testing/training.",
                device_id, device.name,
            )
            self.activation_history.append((device_id, False, "SIMULATED_BACKEND"))
            return False

        # Safety check: quarantined devices
        if device.device_id in QUARANTINED_DEVICES_EXTENDED and not force:
            logger.error(f"REJECTED: Device {device_id} ({device.name}) is QUARANTINED")
            self.activation_history.append((device_id, False, "QUARANTINED"))
            return False

        # Safety check: activation safety
        if not device.safe_to_activate and not force:
            logger.warning(f"REJECTED: Device {device_id} ({device.name}) marked unsafe")
            self.activation_history.append((device_id, False, "UNSAFE"))
            return False

        logger.info(f"Activating device {device_id} ({device.name})...")

        # Try driver activation
        if not self.driver.open():
            logger.error("Failed to open driver")
            return False

        try:
            success = self.driver.activate_device(device_id)

            if success:
                logger.info(f"✓ Device {device_id} ({device.name}) activated successfully")
                self.activation_history.append((device_id, True, "SUCCESS"))
            else:
                logger.error(f"✗ Device {device_id} ({device.name}) activation failed")
                self.activation_history.append((device_id, False, "FAILED"))

            return success

        finally:
            self.driver.close()

    def activate_multiple_devices(self, device_ids: List[int],
                                  progress_callback=None) -> Dict[int, bool]:
        """
        Activate multiple devices in sequence

        Returns dict of {device_id: success}
        """
        results = {}

        for idx, device_id in enumerate(device_ids):
            if progress_callback:
                progress_callback(f"Activating device {device_id} ({idx+1}/{len(device_ids)})...")

            results[device_id] = self.activate_device(device_id)

        return results

    def activate_safe_devices_only(self, progress_callback=None) -> Dict[int, bool]:
        """Activate only devices marked as SAFE"""
        safe_device_ids = [d // 3 for d in SAFE_DEVICES_EXTENDED
                          if d >= 0x8000 and d <= 0x8137]

        logger.info(f"Activating {len(safe_device_ids)} SAFE devices")

        return self.activate_multiple_devices(safe_device_ids, progress_callback)

    # ========================================================================
    # SYSTEM MONITORING
    # ========================================================================

    def get_system_status(self) -> Optional[SystemStatus]:
        """Get comprehensive system status"""
        if not self.driver.open():
            return None

        try:
            return self.driver.get_status()
        finally:
            self.driver.close()

    def get_bios_status(self) -> Optional[BiosStatus]:
        """Get BIOS health and status"""
        if not self.driver.open():
            return None

        try:
            return self.driver.get_bios_status()
        finally:
            self.driver.close()

    def get_thermal_status(self) -> Optional[int]:
        """Get current thermal temperature"""
        status = self.get_system_status()
        return status.thermal_celsius if status else None

    def check_thermal_safe(self, threshold: int = 85) -> bool:
        """Check if thermal levels are safe"""
        temp = self.get_thermal_status()
        if temp is None:
            return False

        return temp < threshold

    # ========================================================================
    # DEVICE QUERIES
    # ========================================================================

    def get_device_status(self, device_id: int) -> Optional[DeviceStatus]:
        """Get cached or live device status"""
        # Return cached status if available
        if device_id in self.device_cache:
            return self.device_cache[device_id]

        # Query live status
        if not self.driver.open():
            return None

        try:
            device_info = self.driver.get_device_info(device_id)
            if device_info:
                self._cache_device_status(device_id, device_info)
                return self.device_cache.get(device_id)
        finally:
            self.driver.close()

        return None

    def get_all_device_statuses(self) -> Dict[int, DeviceStatus]:
        """Get status for all devices in cache"""
        return self.device_cache.copy()

    def is_device_activated(self, device_id: int) -> bool:
        """Check if device is activated"""
        status = self.get_device_status(device_id)
        if not status:
            return False

        return status.activation_status == ActivationStatus.ACTIVATED

    # ========================================================================
    # LEGACY COMPATIBILITY
    # ========================================================================

    def get_legacy_device_count(self) -> int:
        """Get device count (legacy 84-device format)"""
        # Return 84 for compatibility with old tools
        return 84

    def translate_device_id_to_token(self, device_id: int) -> Tuple[int, int, int]:
        """Convert device ID to token IDs (legacy format)"""
        return get_token_range(device_id)

    def get_legacy_controller(self):
        """Get legacy subsystem controller if available"""
        return self.legacy_controller

    # ========================================================================
    # REPORTING
    # ========================================================================

    def generate_discovery_report(self) -> Dict[str, Any]:
        """Generate comprehensive discovery report"""
        stats = get_statistics_extended()

        discovered_ids = list(self.device_cache.keys())
        activated_ids = [d for d, s in self.device_cache.items()
                        if s.activation_status == ActivationStatus.ACTIVATED]

        return {
            'driver_version': self.driver.get_version() if self.driver_available else None,
            'driver_available': self.driver_available,
            'total_devices': 104,
            'devices_in_database': stats['total_devices'],
            'devices_discovered': len(discovered_ids),
            'devices_activated': len(activated_ids),
            'discovered_device_ids': discovered_ids,
            'activated_device_ids': activated_ids,
            'safe_devices': stats['safe'],
            'quarantined_devices': stats['quarantined'],
            'activation_history': self.activation_history,
            'groups': stats['groups'],
        }

    def print_system_summary(self):
        """Print comprehensive system summary"""
        print("=" * 70)
        print("DSMIL INTEGRATION ADAPTER - SYSTEM SUMMARY")
        print("=" * 70)

        # Driver status
        print("\n[Driver Status]")
        print(f"  Available: {self.driver_available}")
        if self.driver_available:
            version = self.driver.get_version()
            print(f"  Version: {version}")

        # System status
        status = self.get_system_status()
        if status:
            print("\n[System Status]")
            print(f"  Devices: {status.device_count}/{104}")
            print(f"  Active BIOS: {chr(ord('A') + status.active_bios)}")
            print(f"  BIOS Health: A={status.bios_health_a} " +
                  f"B={status.bios_health_b} C={status.bios_health_c}")
            print(f"  Thermal: {status.thermal_celsius}°C")
            print(f"  Authenticated: {'Yes' if status.authenticated else 'No'}")

        # Discovery summary
        report = self.generate_discovery_report()
        print("\n[Discovery Summary]")
        print(f"  Discovered: {report['devices_discovered']}/104")
        print(f"  Activated: {report['devices_activated']}")
        print(f"  Safe: {report['safe_devices']}")
        print(f"  Quarantined: {report['quarantined_devices']}")

        # Activation history
        if self.activation_history:
            print("\n[Recent Activations]")
            for device_id, success, msg in self.activation_history[-5:]:
                status_icon = "✓" if success else "✗"
                print(f"  {status_icon} Device {device_id}: {msg}")

        print("\n" + "=" * 70)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_discover() -> DSMILIntegrationAdapter:
    """Quick discovery - returns initialized adapter with devices discovered"""
    adapter = DSMILIntegrationAdapter()

    if adapter.driver_available:
        print("Discovering devices...")
        discovered = adapter.discover_all_devices_cascading(
            progress_callback=lambda msg: print(f"  {msg}")
        )
        print(f"✓ Discovered {len(discovered)} devices")

    return adapter


def activate_all_safe_devices() -> DSMILIntegrationAdapter:
    """Convenience function to activate all safe devices"""
    adapter = quick_discover()

    print("\nActivating all SAFE devices...")
    results = adapter.activate_safe_devices_only(
        progress_callback=lambda msg: print(f"  {msg}")
    )

    success_count = sum(1 for r in results.values() if r)
    print(f"✓ Activated {success_count}/{len(results)} safe devices")

    return adapter


# ============================================================================
# MAIN - INTEGRATION TEST
# ============================================================================

def main():
    """Integration adapter test and demonstration"""
    print("=" * 70)
    print("DSMIL INTEGRATION ADAPTER")
    print("Unified Interface for 104-Device Driver")
    print("=" * 70)
    print()

    # Initialize adapter
    print("[1/4] Initializing adapter...")
    adapter = DSMILIntegrationAdapter()

    if not adapter.driver_available:
        print("  ✗ Driver not available")
        print("  Run: sudo insmod 01-source/kernel/core/dsmil-104dev.ko")
        return 1

    print("  ✓ Adapter initialized")

    # Discover devices
    print("\n[2/4] Discovering devices...")
    discovered = adapter.discover_all_devices_cascading(
        progress_callback=lambda msg: print(f"  {msg}")
    )
    print(f"  ✓ Discovered {len(discovered)} devices")

    # Show system status
    print("\n[3/4] System status...")
    adapter.print_system_summary()

    # Activate safe devices
    print("\n[4/4] Activating safe devices...")
    results = adapter.activate_safe_devices_only(
        progress_callback=lambda msg: print(f"  {msg}")
    )

    success_count = sum(1 for r in results.values() if r)
    print(f"  ✓ Activated {success_count}/{len(results)} safe devices")

    # Final report
    print("\n" + "=" * 70)
    print("INTEGRATION TEST COMPLETE")
    print("=" * 70)

    report = adapter.generate_discovery_report()
    print(f"Devices Discovered: {report['devices_discovered']}/104")
    print(f"Devices Activated: {report['devices_activated']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
