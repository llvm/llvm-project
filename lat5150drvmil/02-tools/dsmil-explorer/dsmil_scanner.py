#!/usr/bin/env python3
"""
DSMIL Device Capability Scanner

Fast scanning tool for enumerating DSMIL devices and their capabilities.
Provides quick overview of device landscape without deep probing.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Optional

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dsmil_safety import SafetyValidator, DeviceRisk, SAFE_DEVICES, RISKY_DEVICES, QUARANTINED_DEVICES
from dsmil_common import DSMILDevice, DeviceAccess, get_device_list
from dsmil_logger import create_logger, LogLevel

class DeviceInfo:
    """Device information summary"""

    def __init__(self, device_id: int):
        self.device_id = device_id
        self.name = None
        self.risk_level = DeviceRisk.UNKNOWN
        self.accessible = False
        self.capabilities = None
        self.status = None
        self.group = (device_id - 0x8000) // 12
        self.index_in_group = (device_id - 0x8000) % 12

    def to_dict(self) -> Dict:
        return {
            "device_id": f"0x{self.device_id:04X}",
            "name": self.name,
            "risk_level": self.risk_level.value if self.risk_level else "UNKNOWN",
            "group": self.group,
            "index_in_group": self.index_in_group,
            "accessible": self.accessible,
            "capabilities": self.capabilities,
            "status": self.status,
        }

class DeviceScanner:
    """Fast device capability scanner"""

    def __init__(self, logger=None, safety=None):
        self.logger = logger
        self.safety = safety or SafetyValidator()
        self.scanned_devices = {}

    def scan_device(self, device_id: int, quick: bool = False) -> DeviceInfo:
        """
        Scan a single device

        Args:
            device_id: Device to scan
            quick: Quick scan (no hardware access)

        Returns:
            DeviceInfo with collected data
        """
        info = DeviceInfo(device_id)

        # Get risk level
        info.risk_level = self.safety.get_device_risk_level(device_id)

        # Set name if known
        if device_id in SAFE_DEVICES:
            info.name = SAFE_DEVICES[device_id]
        elif device_id in RISKY_DEVICES:
            info.name = RISKY_DEVICES[device_id]
        elif device_id in QUARANTINED_DEVICES:
            info.name = QUARANTINED_DEVICES[device_id]

        # Check if accessible
        allowed, reason, level = self.safety.check_device_access(device_id)
        info.accessible = allowed

        # Hardware access (if not quick scan and device is accessible)
        if not quick and allowed and info.risk_level != DeviceRisk.QUARANTINED:
            try:
                with DeviceAccess() as dev:
                    if dev.is_open:
                        info.capabilities = dev.get_device_capabilities(device_id)
                        info.status = dev.get_device_status(device_id)
            except Exception as e:
                if self.logger:
                    self.logger.warning("scanner", f"Failed to access device: {e}",
                                       device_id=device_id)

        self.scanned_devices[device_id] = info
        return info

    def scan_all(self, quick: bool = False) -> Dict[int, DeviceInfo]:
        """Scan all DSMIL devices"""
        if self.logger:
            self.logger.info("scanner", f"Starting full scan (quick={quick})")

        devices = get_device_list()

        for device in devices:
            self.scan_device(device.device_id, quick=quick)

            if not quick:
                time.sleep(0.1)  # Small delay between devices

        if self.logger:
            self.logger.info("scanner", f"Scan complete: {len(self.scanned_devices)} devices")

        return self.scanned_devices

    def scan_group(self, group: int, quick: bool = False) -> Dict[int, DeviceInfo]:
        """Scan devices in a specific group (0-6)"""
        if group < 0 or group > 6:
            raise ValueError(f"Invalid group: {group} (must be 0-6)")

        start_id = 0x8000 + (group * 12)
        end_id = start_id + 11

        if self.logger:
            self.logger.info("scanner", f"Scanning group {group}: 0x{start_id:04X}-0x{end_id:04X}")

        results = {}
        for device_id in range(start_id, end_id + 1):
            info = self.scan_device(device_id, quick=quick)
            results[device_id] = info

            if not quick:
                time.sleep(0.1)

        return results

    def get_summary(self) -> Dict:
        """Get scan summary statistics"""
        total = len(self.scanned_devices)
        by_risk = {}
        accessible = 0

        for info in self.scanned_devices.values():
            risk_name = info.risk_level.value if info.risk_level else "UNKNOWN"
            by_risk[risk_name] = by_risk.get(risk_name, 0) + 1

            if info.accessible:
                accessible += 1

        return {
            "total_devices": total,
            "accessible": accessible,
            "by_risk_level": by_risk,
        }

    def print_summary(self):
        """Print formatted scan summary"""
        summary = self.get_summary()

        print("\n" + "=" * 80)
        print("DSMIL Device Scan Summary")
        print("=" * 80)
        print(f"Total devices: {summary['total_devices']}")
        print(f"Accessible: {summary['accessible']}")
        print("\nBy Risk Level:")
        for risk, count in sorted(summary['by_risk_level'].items()):
            percentage = (count / summary['total_devices']) * 100
            print(f"  {risk:15} {count:3} devices ({percentage:5.1f}%)")

    def print_device_table(self, group: Optional[int] = None):
        """Print formatted device table"""
        print("\n" + "=" * 120)
        if group is not None:
            print(f"DSMIL Devices - Group {group}")
        else:
            print("DSMIL Devices - All Groups")
        print("=" * 120)
        print(f"{'Device ID':10} {'Name':25} {'Group':7} {'Risk':15} {'Access':8} {'Status':10}")
        print("-" * 120)

        devices_to_show = self.scanned_devices.values()
        if group is not None:
            devices_to_show = [d for d in devices_to_show if d.group == group]

        for info in sorted(devices_to_show, key=lambda x: x.device_id):
            device_id_str = f"0x{info.device_id:04X}"
            name_str = (info.name or "Unknown")[:24]
            group_str = f"G{info.group}.{info.index_in_group}"
            risk_str = info.risk_level.value if info.risk_level else "UNKNOWN"
            access_str = "✓ Yes" if info.accessible else "✗ No"

            # Determine status
            if info.risk_level == DeviceRisk.QUARANTINED:
                status_str = "BLOCKED"
            elif info.status and info.status.get("ready"):
                status_str = "Ready"
            elif info.accessible:
                status_str = "Unknown"
            else:
                status_str = "N/A"

            print(f"{device_id_str:10} {name_str:25} {group_str:7} {risk_str:15} {access_str:8} {status_str:10}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Device Capability Scanner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick scan (no hardware access)
  python3 dsmil_scanner.py --quick

  # Full scan with hardware probing
  sudo python3 dsmil_scanner.py

  # Scan specific group
  sudo python3 dsmil_scanner.py --group 3

  # Export to JSON
  python3 dsmil_scanner.py --quick --export devices.json
        """
    )

    parser.add_argument('--quick', action='store_true',
                       help='Quick scan (no hardware access)')
    parser.add_argument('--group', type=int, choices=range(7), metavar='N',
                       help='Scan specific group (0-6)')
    parser.add_argument('--export', metavar='FILE',
                       help='Export results to JSON file')
    parser.add_argument('--log-dir', default='output/scan_logs',
                       help='Log output directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    # Create logger
    log_level = LogLevel.DEBUG if args.verbose else LogLevel.INFO
    logger = create_logger(log_dir=args.log_dir, min_level=log_level)

    # Create safety validator
    safety = SafetyValidator()

    # Create scanner
    scanner = DeviceScanner(logger=logger, safety=safety)

    try:
        # Perform scan
        if args.group is not None:
            results = scanner.scan_group(args.group, quick=args.quick)
        else:
            results = scanner.scan_all(quick=args.quick)

        # Print results
        scanner.print_summary()
        scanner.print_device_table(group=args.group)

        # Export if requested
        if args.export:
            export_data = {
                "timestamp": time.time(),
                "scan_type": "quick" if args.quick else "full",
                "group": args.group,
                "summary": scanner.get_summary(),
                "devices": {f"0x{k:04X}": v.to_dict() for k, v in results.items()},
            }

            with open(args.export, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"\nResults exported to: {args.export}")

    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")

    except Exception as e:
        print(f"\nError: {e}")
        logger.error("scanner", f"Fatal error: {e}")
        return 1

    finally:
        stats = logger.get_statistics()
        print(f"\nLog file: {stats['log_file']}")
        logger.close()

    return 0

if __name__ == "__main__":
    sys.exit(main())
