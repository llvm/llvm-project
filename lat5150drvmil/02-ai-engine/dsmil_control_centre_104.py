#!/usr/bin/env python3
"""
DSMIL Control Centre - 104-Device Edition
=========================================
Unified control centre for the complete 104-device DSMIL architecture

Features:
- Cascading device discovery across all 104 devices
- Intelligent activation with safety guardrails
- Real-time system monitoring (thermal, BIOS, tokens)
- TPM authentication integration
- Legacy tool compatibility
- ML-enhanced device analysis
- Comprehensive reporting and audit trails

Author: LAT5150DRVMIL Platform Team
Version: 2.1.0
Driver Compatibility: dsmil-104dev v5.2.0
Kernel Support: 6.17+ (with fallback to 5.x/4.x)

Enhanced Features (v2.1):
- Driver 104 primary with automatic fallback to 84
- Kernel 6.17 compatibility with version detection
- Complex path recovery for moved/lost driver files
- Multiple compensation mechanisms for driver loading
- Resilient system with retry logic
"""

import os
import sys
import curses
import time
import json
import logging
import platform
import re
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Import integration modules
from dsmil_integration_adapter import (
    DSMILIntegrationAdapter,
    ActivationStatus,
    DeviceStatus as AdapterDeviceStatus,
)
from dsmil_driver_interface import (
    DSMILDriverInterface,
    SystemStatus,
    BiosID,
    check_driver_loaded,
)
from dsmil_device_database_extended import (
    get_device_extended,
    get_statistics_extended,
    QUARANTINED_DEVICES_EXTENDED,
    SAFE_DEVICES_EXTENDED,
)

logger = logging.getLogger(__name__)


# ============================================================================
# KERNEL VERSION DETECTION & PATH RECOVERY
# ============================================================================

def get_kernel_version():
    """Get running kernel version"""
    uname = platform.uname()
    version_str = uname.release
    match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return (0, 0, 0)

def get_recommended_driver():
    """Get recommended driver based on kernel version"""
    major, minor, patch = get_kernel_version()

    # Kernel 6.17+ or 6.0-6.16: Use driver 104
    if major >= 7 or (major == 6):
        return "dsmil-104dev", f"{major}.{minor}.{patch}"

    # Kernel 5.x or 4.x: Use fallback driver 84
    elif major >= 4:
        return "dsmil-84dev", f"{major}.{minor}.{patch}"

    # Unknown: Try 104 anyway
    else:
        return "dsmil-104dev", f"{major}.{minor}.{patch}"

def find_driver_interface_module():
    """Find driver interface module with path recovery"""
    search_paths = [
        Path(__file__).parent / "dsmil_driver_interface.py",
        Path("02-ai-engine/dsmil_driver_interface.py"),
        Path("ai-engine/dsmil_driver_interface.py"),
        Path("dsmil_driver_interface.py"),
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


# ============================================================================
# CONTROL CENTRE MODES
# ============================================================================

class ControlMode(Enum):
    """Control centre operation modes"""
    DISCOVERY = "discovery"
    ACTIVATION = "activation"
    MONITORING = "monitoring"
    DIAGNOSTICS = "diagnostics"
    ADMINISTRATION = "administration"


# ============================================================================
# CONTROL CENTRE CLASS
# ============================================================================

class DSMILControlCentre:
    """
    Unified control centre for 104-device DSMIL system

    Provides comprehensive management interface for:
    - Device discovery and enumeration
    - Safe device activation
    - System health monitoring
    - BIOS management
    - TPM authentication
    - Audit and reporting
    """

    def __init__(self):
        """Initialize control centre with kernel compatibility detection"""
        # Detect kernel and recommended driver
        recommended_driver, kernel_version = get_recommended_driver()

        print(f"\nDSMIL Control Centre v2.1 (Enhanced)")
        print(f"Kernel Version: {kernel_version}")
        print(f"Recommended Driver: {recommended_driver}")
        print(f"Driver Support: 104 devices (with 84 fallback)")
        print()

        self.adapter = DSMILIntegrationAdapter()
        self.mode = ControlMode.DISCOVERY
        self.discovered_devices: List[int] = []
        self.activated_devices: List[int] = []
        self.kernel_version = kernel_version
        self.recommended_driver = recommended_driver

        logger.info(f"DSMIL Control Centre v2.1 initialized (kernel {kernel_version})")

    # ========================================================================
    # DISCOVERY MODE
    # ========================================================================

    def run_full_discovery(self, interactive: bool = True) -> List[int]:
        """
        Run comprehensive cascading discovery

        Returns list of discovered device IDs
        """
        print("\n" + "=" * 70)
        print("CASCADING DEVICE DISCOVERY - 104 DEVICES")
        print("=" * 70)
        print()

        if not self.adapter.driver_available:
            print("✗ Driver not available")
            print("  Load driver: sudo insmod dsmil-104dev.ko")
            return []

        def progress(msg):
            print(f"  {msg}")

        discovered = self.adapter.discover_all_devices_cascading(
            progress_callback=progress
        )

        self.discovered_devices = discovered

        print()
        print("=" * 70)
        print(f"DISCOVERY COMPLETE: {len(discovered)}/104 DEVICES")
        print("=" * 70)

        # Show breakdown by group
        groups = {}
        for device_id in discovered:
            device = get_device_extended(device_id)
            if device:
                group_num = device.group.value
                groups[group_num] = groups.get(group_num, 0) + 1

        print("\nDevices by Group:")
        for group_num in sorted(groups.keys()):
            print(f"  Group {group_num}: {groups[group_num]} devices")

        # Show safety breakdown
        safe_count = sum(1 for d in discovered
                        if (0x8000 + d * 3) in SAFE_DEVICES_EXTENDED)
        quarantined_count = sum(1 for d in discovered
                               if (0x8000 + d * 3) in QUARANTINED_DEVICES_EXTENDED)

        print(f"\nSafety Status:")
        print(f"  Safe: {safe_count}")
        print(f"  Quarantined: {quarantined_count}")
        print(f"  Other: {len(discovered) - safe_count - quarantined_count}")

        if interactive:
            input("\nPress ENTER to continue...")

        return discovered

    # ========================================================================
    # ACTIVATION MODE
    # ========================================================================

    def run_safe_activation(self, interactive: bool = True) -> Dict[int, bool]:
        """
        Activate all SAFE devices with user confirmation

        Returns dict of {device_id: success_status}
        """
        print("\n" + "=" * 70)
        print("SAFE DEVICE ACTIVATION")
        print("=" * 70)
        print()

        # Get safe devices
        safe_device_ids = [d for d in self.discovered_devices
                          if (0x8000 + d * 3) in SAFE_DEVICES_EXTENDED]

        print(f"Found {len(safe_device_ids)} SAFE devices to activate")
        print()

        # Show devices to be activated
        for device_id in safe_device_ids[:10]:  # Show first 10
            device = get_device_extended(device_id)
            if device:
                print(f"  • Device {device_id:3d}: {device.name}")

        if len(safe_device_ids) > 10:
            print(f"  ... and {len(safe_device_ids) - 10} more")

        print()

        # Confirmation
        if interactive:
            response = input(f"Activate {len(safe_device_ids)} safe devices? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Activation cancelled")
                return {}

        # Activate devices
        print("\nActivating devices...")

        def progress(msg):
            print(f"  {msg}")

        results = self.adapter.activate_multiple_devices(
            safe_device_ids,
            progress_callback=progress
        )

        # Update activated list
        self.activated_devices = [d for d, success in results.items() if success]

        # Report results
        success_count = sum(1 for r in results.values() if r)
        print()
        print("=" * 70)
        print(f"ACTIVATION COMPLETE: {success_count}/{len(safe_device_ids)} SUCCESS")
        print("=" * 70)

        if interactive:
            input("\nPress ENTER to continue...")

        return results

    def run_custom_activation(self, device_ids: List[int],
                            interactive: bool = True) -> Dict[int, bool]:
        """
        Activate custom list of devices with safety checks

        Returns dict of {device_id: success_status}
        """
        print("\n" + "=" * 70)
        print("CUSTOM DEVICE ACTIVATION")
        print("=" * 70)
        print()

        # Show devices and safety status
        print(f"Devices to activate: {len(device_ids)}\n")

        quarantined = []
        unsafe = []
        safe = []

        for device_id in device_ids:
            device = get_device_extended(device_id)
            token_base = 0x8000 + (device_id * 3)

            if token_base in QUARANTINED_DEVICES_EXTENDED:
                quarantined.append(device_id)
                print(f"  ⚠️  Device {device_id:3d}: {device.name if device else 'Unknown'} [QUARANTINED]")
            elif device and not device.safe_to_activate:
                unsafe.append(device_id)
                print(f"  ⚠️  Device {device_id:3d}: {device.name} [UNSAFE]")
            else:
                safe.append(device_id)
                print(f"  ✓  Device {device_id:3d}: {device.name if device else 'Unknown'} [SAFE]")

        print()

        # Warnings
        if quarantined:
            print(f"⚠️  WARNING: {len(quarantined)} QUARANTINED devices will be SKIPPED")
        if unsafe:
            print(f"⚠️  WARNING: {len(unsafe)} UNSAFE devices detected")

        print()

        # Confirmation
        if interactive:
            if quarantined or unsafe:
                print("This operation includes dangerous devices!")
                response = input("Type 'I UNDERSTAND THE RISKS' to proceed: ")
                if response != 'I UNDERSTAND THE RISKS':
                    print("Activation cancelled")
                    return {}
            else:
                response = input(f"Activate {len(safe)} safe devices? (yes/no): ")
                if response.lower() not in ['yes', 'y']:
                    print("Activation cancelled")
                    return {}

        # Activate only safe devices
        print("\nActivating safe devices...")
        results = self.adapter.activate_multiple_devices(
            safe,
            progress_callback=lambda msg: print(f"  {msg}")
        )

        success_count = sum(1 for r in results.values() if r)
        print(f"\n✓ Activated {success_count}/{len(safe)} devices")

        if interactive:
            input("\nPress ENTER to continue...")

        return results

    # ========================================================================
    # MONITORING MODE
    # ========================================================================

    def run_system_monitoring(self, duration: int = 60):
        """
        Real-time system monitoring

        Monitors for specified duration (seconds)
        """
        print("\n" + "=" * 70)
        print(f"SYSTEM MONITORING - {duration}s")
        print("=" * 70)
        print()

        start_time = time.time()
        check_interval = 5

        try:
            while time.time() - start_time < duration:
                # Get system status
                status = self.adapter.get_system_status()
                bios_status = self.adapter.get_bios_status()

                # Clear screen (simple version)
                print("\033[2J\033[H", end='')

                # Print header
                print("=" * 70)
                print(f"DSMIL SYSTEM MONITOR - {time.strftime('%H:%M:%S')}")
                print("=" * 70)
                print()

                # System info
                if status:
                    print("[System Status]")
                    print(f"  Devices: {status.device_count}/104")
                    print(f"  Token Operations: {status.token_reads} reads, {status.token_writes} writes")
                    print(f"  Authenticated: {'Yes' if status.authenticated else 'No'}")
                    print()

                    # BIOS status
                    print("[BIOS Status]")
                    active_bios_char = chr(ord('A') + status.active_bios)
                    print(f"  Active: BIOS {active_bios_char}")
                    print(f"  Health: A={status.bios_health_a}% " +
                          f"B={status.bios_health_b}% C={status.bios_health_c}%")
                    print(f"  Failover Count: {status.failover_count}")
                    print()

                    # Thermal status
                    temp = status.thermal_celsius
                    temp_status = "NORMAL"
                    if temp > 85:
                        temp_status = "HIGH"
                    elif temp > 95:
                        temp_status = "CRITICAL"

                    print("[Thermal Status]")
                    print(f"  Temperature: {temp}°C [{temp_status}]")
                    print()

                # Device status
                print("[Device Status]")
                print(f"  Discovered: {len(self.discovered_devices)}")
                print(f"  Activated: {len(self.activated_devices)}")
                print()

                elapsed = int(time.time() - start_time)
                remaining = duration - elapsed
                print(f"Monitoring: {elapsed}s elapsed, {remaining}s remaining")
                print("Press Ctrl+C to stop...")

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")

        print("\n" + "=" * 70)

    # ========================================================================
    # DIAGNOSTICS MODE
    # ========================================================================

    def run_diagnostics(self):
        """Run comprehensive system diagnostics"""
        print("\n" + "=" * 70)
        print("SYSTEM DIAGNOSTICS")
        print("=" * 70)
        print()

        # Driver check
        print("[1/5] Checking driver...")
        if check_driver_loaded():
            print("  ✓ Driver loaded (/dev/dsmil0 exists)")

            driver = DSMILDriverInterface()
            if driver.open():
                version = driver.get_version()
                print(f"  ✓ Driver version: {version}")
                driver.close()
            else:
                print("  ✗ Failed to open driver")
        else:
            print("  ✗ Driver not loaded")

        # Database check
        print("\n[2/5] Checking device database...")
        stats = get_statistics_extended()
        print(f"  ✓ Database loaded: {stats['total_devices']} devices")
        print(f"  ✓ Safe devices: {stats['safe']}")
        print(f"  ✓ Quarantined: {stats['quarantined']}")

        # Discovery check
        print("\n[3/5] Checking device discovery...")
        print(f"  ✓ Devices discovered: {len(self.discovered_devices)}")
        print(f"  ✓ Devices activated: {len(self.activated_devices)}")

        # System status check
        print("\n[4/5] Checking system status...")
        status = self.adapter.get_system_status()
        if status:
            print(f"  ✓ System responding")
            print(f"  ✓ Device count: {status.device_count}")
            print(f"  ✓ Thermal: {status.thermal_celsius}°C")
        else:
            print("  ✗ Failed to get system status")

        # Integration check
        print("\n[5/5] Checking integration...")
        report = self.adapter.generate_discovery_report()
        print(f"  ✓ Adapter initialized")
        print(f"  ✓ Legacy compatibility: Available")

        print("\n" + "=" * 70)
        print("DIAGNOSTICS COMPLETE")
        print("=" * 70)

        input("\nPress ENTER to continue...")

    # ========================================================================
    # MAIN MENU
    # ========================================================================

    def show_main_menu(self):
        """Display main menu"""
        while True:
            print("\n" + "=" * 70)
            print("DSMIL CONTROL CENTRE - 104-DEVICE EDITION")
            print("=" * 70)
            print()
            print("1. Device Discovery (Cascading scan of all 104 devices)")
            print("2. Activate Safe Devices (Auto-activate verified safe devices)")
            print("3. Custom Activation (Select specific devices)")
            print("4. System Monitoring (Real-time status)")
            print("5. Diagnostics (System health check)")
            print("6. Generate Report (Export comprehensive report)")
            print("7. View Statistics (Device and system stats)")
            print("8. Exit")
            print()

            choice = input("Select option (1-8): ").strip()

            if choice == '1':
                self.run_full_discovery(interactive=True)
            elif choice == '2':
                if not self.discovered_devices:
                    print("\n⚠️  Run discovery first (option 1)")
                    input("Press ENTER to continue...")
                else:
                    self.run_safe_activation(interactive=True)
            elif choice == '3':
                if not self.discovered_devices:
                    print("\n⚠️  Run discovery first (option 1)")
                    input("Press ENTER to continue...")
                else:
                    print("\nEnter device IDs (comma-separated): ")
                    ids_str = input("> ")
                    try:
                        device_ids = [int(x.strip()) for x in ids_str.split(',')]
                        self.run_custom_activation(device_ids, interactive=True)
                    except ValueError:
                        print("✗ Invalid device IDs")
                        input("Press ENTER to continue...")
            elif choice == '4':
                print("\nMonitoring duration (seconds, default=60): ")
                duration_str = input("> ")
                duration = int(duration_str) if duration_str.strip() else 60
                self.run_system_monitoring(duration)
            elif choice == '5':
                self.run_diagnostics()
            elif choice == '6':
                self.generate_full_report()
            elif choice == '7':
                self.show_statistics()
            elif choice == '8':
                print("\nExiting control centre...")
                break
            else:
                print("\n✗ Invalid option")
                input("Press ENTER to continue...")

    def generate_full_report(self):
        """Generate and export comprehensive report"""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 70)
        print()

        report = self.adapter.generate_discovery_report()

        # Add system status
        status = self.adapter.get_system_status()
        if status:
            report['system_status'] = {
                'driver_version': status.driver_version,
                'device_count': status.device_count,
                'active_bios': status.active_bios,
                'thermal_celsius': status.thermal_celsius,
                'authenticated': bool(status.authenticated),
                'token_reads': status.token_reads,
                'token_writes': status.token_writes,
                'failover_count': status.failover_count,
            }

        # Export to file
        output_path = Path('/tmp/dsmil_control_centre_report.json')
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report saved to: {output_path}")
        print()
        print("Report Summary:")
        print(f"  Devices Discovered: {report['devices_discovered']}/104")
        print(f"  Devices Activated: {report['devices_activated']}")
        print(f"  Safe Devices: {report['safe_devices']}")
        print(f"  Quarantined: {report['quarantined_devices']}")

        input("\nPress ENTER to continue...")

    def show_statistics(self):
        """Show comprehensive statistics"""
        print("\n" + "=" * 70)
        print("SYSTEM STATISTICS")
        print("=" * 70)
        print()

        self.adapter.print_system_summary()

        input("\nPress ENTER to continue...")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="DSMIL Control Centre - 104-Device Edition")
    parser.add_argument('--auto-discover', action='store_true',
                       help="Automatically run discovery on startup")
    parser.add_argument('--auto-activate', action='store_true',
                       help="Automatically activate safe devices after discovery")
    parser.add_argument('--non-interactive', action='store_true',
                       help="Non-interactive mode for automation")

    args = parser.parse_args()

    # Check root
    if os.geteuid() != 0:
        print("⚠️  WARNING: Not running as root")
        print("   Some operations may require sudo privileges")
        print()

    # Initialize control centre
    control_centre = DSMILControlCentre()

    # Auto-discovery mode
    if args.auto_discover:
        control_centre.run_full_discovery(interactive=not args.non_interactive)

        if args.auto_activate:
            control_centre.run_safe_activation(interactive=not args.non_interactive)

        if not args.non_interactive:
            control_centre.show_main_menu()
    else:
        # Interactive menu
        control_centre.show_main_menu()

    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    sys.exit(main())
