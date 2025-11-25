#!/usr/bin/env python3
"""
DSMIL Auto-Documentation Generator

Automatically generates comprehensive documentation from device probe results,
including device profiles, API documentation, and integration guides.

Author: DSMIL Automation Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dsmil_safety import SAFE_DEVICES, RISKY_DEVICES, QUARANTINED_DEVICES, DeviceRisk

class DocumentationGenerator:
    """Auto-documentation generator"""

    def __init__(self, output_dir="output/docs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_device_profile(self, device_id: int, probe_data: Dict, output_file: str = None):
        """Generate device profile documentation"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"device_0x{device_id:04X}_profile.md")

        # Get device info
        device_name = self._get_device_name(device_id)
        risk_level = self._get_risk_level(device_id)
        group = (device_id - 0x8000) // 12

        # Build markdown
        md = []
        md.append(f"# Device Profile: 0x{device_id:04X}")
        md.append("")
        md.append(f"**Device Name**: {device_name}")
        md.append(f"**Device ID**: 0x{device_id:04X}")
        md.append(f"**Group**: {group}")
        md.append(f"**Risk Level**: {risk_level.value}")
        md.append(f"**Generated**: {datetime.now().isoformat()}")
        md.append("")

        # Safety information
        md.append("## Safety Information")
        md.append("")
        if risk_level == DeviceRisk.QUARANTINED:
            md.append("⛔ **CRITICAL WARNING**: This device is permanently QUARANTINED")
            md.append("")
            md.append("This device performs destructive operations and must NEVER be accessed.")
        elif risk_level == DeviceRisk.RISKY:
            md.append("⚠️  **CAUTION**: This device is classified as RISKY")
            md.append("")
            md.append("Extra validation required before operations.")
        elif risk_level == DeviceRisk.MONITORED:
            md.append("✅ **SAFE**: This device is monitored and safe for read operations")
        else:
            md.append("❓ **UNKNOWN**: This device requires exploration")
        md.append("")

        # Capabilities
        if probe_data.get("capabilities"):
            md.append("## Device Capabilities")
            md.append("")
            md.append("```json")
            md.append(json.dumps(probe_data["capabilities"], indent=2))
            md.append("```")
            md.append("")

        # Probe results
        if probe_data.get("phases_completed"):
            md.append("## Probe Results")
            md.append("")
            md.append(f"**Phases Completed**: {', '.join(probe_data['phases_completed'])}")
            md.append(f"**Success**: {'Yes' if probe_data.get('success') else 'No'}")
            md.append("")

        # Observations
        if probe_data.get("observations"):
            md.append("## Observations")
            md.append("")
            obs = probe_data["observations"]
            if "register_values" in obs:
                md.append("### Register Values")
                md.append("")
                md.append("| Offset | Value |")
                md.append("|--------|-------|")
                for i, val in enumerate(obs["register_values"]):
                    md.append(f"| 0x{i*4:02X} | {val} |")
                md.append("")

        # Errors and warnings
        if probe_data.get("errors"):
            md.append("## Errors")
            md.append("")
            for error in probe_data["errors"]:
                md.append(f"- {error}")
            md.append("")

        if probe_data.get("warnings"):
            md.append("## Warnings")
            md.append("")
            for warning in probe_data["warnings"]:
                md.append(f"- {warning}")
            md.append("")

        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(md))

        return output_file

    def generate_group_summary(self, group: int, devices_data: Dict[int, Dict], output_file: str = None):
        """Generate summary documentation for a device group"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, f"group_{group}_summary.md")

        group_names = [
            "Core Security & Emergency",
            "Extended Security",
            "Network & Communications",
            "Data Processing",
            "Storage Control",
            "Peripheral Management",
            "Training Functions"
        ]

        md = []
        md.append(f"# DSMIL Device Group {group}: {group_names[group]}")
        md.append("")
        md.append(f"**Device Range**: 0x{0x8000 + group*12:04X} - 0x{0x8000 + group*12 + 11:04X}")
        md.append(f"**Total Devices**: 12")
        md.append(f"**Generated**: {datetime.now().isoformat()}")
        md.append("")

        # Summary table
        md.append("## Device Summary")
        md.append("")
        md.append("| Device ID | Name | Risk | Status | Explored |")
        md.append("|-----------|------|------|--------|----------|")

        for i in range(12):
            device_id = 0x8000 + (group * 12) + i
            name = self._get_device_name(device_id)
            risk = self._get_risk_level(device_id).value
            
            data = devices_data.get(device_id, {})
            status = "✓ Success" if data.get("success") else ("✗ Failed" if data else "Not probed")
            explored = "Yes" if data.get("phases_completed") else "No"

            md.append(f"| 0x{device_id:04X} | {name:20} | {risk:12} | {status:10} | {explored:8} |")

        md.append("")

        # Statistics
        total = len(devices_data)
        successful = sum(1 for d in devices_data.values() if d.get("success"))

        md.append("## Statistics")
        md.append("")
        md.append(f"- **Devices Probed**: {total}/12")
        md.append(f"- **Successful**: {successful}/{total}" if total > 0 else "- **Successful**: 0/0")
        md.append(f"- **Coverage**: {(total/12)*100:.1f}%")
        md.append("")

        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(md))

        return output_file

    def generate_master_index(self, all_devices_data: Dict[int, Dict], output_file: str = None):
        """Generate master index of all devices"""
        if output_file is None:
            output_file = os.path.join(self.output_dir, "DEVICE_INDEX.md")

        md = []
        md.append("# DSMIL Device Master Index")
        md.append("")
        md.append(f"**Total Devices**: 84 (0x8000 - 0x806B)")
        md.append(f"**Generated**: {datetime.now().isoformat()}")
        md.append("")

        # Overall statistics
        total_explored = len(all_devices_data)
        total_successful = sum(1 for d in all_devices_data.values() if d.get("success"))

        md.append("## Overall Statistics")
        md.append("")
        md.append(f"- **Devices Explored**: {total_explored}/84 ({(total_explored/84)*100:.1f}%)")
        md.append(f"- **Successful Probes**: {total_successful}/{total_explored}" if total_explored > 0 else "- **Successful Probes**: 0/0")
        md.append(f"- **Known Safe**: {len(SAFE_DEVICES)}")
        md.append(f"- **Known Risky**: {len(RISKY_DEVICES)}")
        md.append(f"- **Quarantined**: {len(QUARANTINED_DEVICES)}")
        md.append("")

        # Per-group summary
        md.append("## Device Groups")
        md.append("")

        group_names = [
            "Core Security & Emergency",
            "Extended Security",
            "Network & Communications",
            "Data Processing",
            "Storage Control",
            "Peripheral Management",
            "Training Functions"
        ]

        for group in range(7):
            start_id = 0x8000 + (group * 12)
            end_id = start_id + 11

            # Count devices in this group
            group_devices = [d for d in all_devices_data.keys() if start_id <= d <= end_id]
            group_successful = sum(1 for did in group_devices if all_devices_data[did].get("success"))

            md.append(f"### Group {group}: {group_names[group]}")
            md.append(f"**Range**: 0x{start_id:04X} - 0x{end_id:04X}")
            md.append(f"**Explored**: {len(group_devices)}/12 ({(len(group_devices)/12)*100:.1f}%)")
            md.append(f"**Successful**: {group_successful}/{len(group_devices)}" if group_devices else "**Successful**: 0/0")
            md.append("")

        # Write file
        with open(output_file, 'w') as f:
            f.write('\n'.join(md))

        return output_file

    def _get_device_name(self, device_id: int) -> str:
        """Get device name"""
        if device_id in SAFE_DEVICES:
            return SAFE_DEVICES[device_id]
        elif device_id in RISKY_DEVICES:
            return RISKY_DEVICES[device_id]
        elif device_id in QUARANTINED_DEVICES:
            return QUARANTINED_DEVICES[device_id]
        else:
            return "Unknown"

    def _get_risk_level(self, device_id: int) -> DeviceRisk:
        """Get device risk level"""
        if device_id in QUARANTINED_DEVICES:
            return DeviceRisk.QUARANTINED
        elif device_id in SAFE_DEVICES:
            return DeviceRisk.MONITORED
        elif device_id in RISKY_DEVICES:
            return DeviceRisk.RISKY
        else:
            return DeviceRisk.UNKNOWN

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="DSMIL Auto-Documentation Generator"
    )

    parser.add_argument('--input', required=True,
                       help='Input JSON file with probe results')
    parser.add_argument('--output-dir', default='output/docs',
                       help='Output directory for documentation')
    parser.add_argument('--device', type=lambda x: int(x, 0),
                       help='Generate profile for specific device')
    parser.add_argument('--group', type=int, choices=range(7),
                       help='Generate summary for specific group')
    parser.add_argument('--index', action='store_true',
                       help='Generate master index')

    args = parser.parse_args()

    # Load probe data
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading input file: {e}")
        return 1

    # Create generator
    docgen = DocumentationGenerator(output_dir=args.output_dir)

    try:
        devices_data = data.get("devices", {})

        # Convert keys to integers
        devices_data = {int(k, 16): v for k, v in devices_data.items()}

        if args.device:
            # Generate device profile
            if args.device in devices_data:
                output_file = docgen.generate_device_profile(args.device, devices_data[args.device])
                print(f"Generated device profile: {output_file}")
            else:
                print(f"No data found for device 0x{args.device:04X}")

        elif args.group is not None:
            # Generate group summary
            start_id = 0x8000 + (args.group * 12)
            end_id = start_id + 11
            group_devices = {k: v for k, v in devices_data.items() if start_id <= k <= end_id}

            output_file = docgen.generate_group_summary(args.group, group_devices)
            print(f"Generated group summary: {output_file}")

        elif args.index:
            # Generate master index
            output_file = docgen.generate_master_index(devices_data)
            print(f"Generated master index: {output_file}")

        else:
            # Generate all
            print("Generating comprehensive documentation...")

            # Master index
            index_file = docgen.generate_master_index(devices_data)
            print(f"  Master index: {index_file}")

            # Group summaries
            for group in range(7):
                start_id = 0x8000 + (group * 12)
                end_id = start_id + 11
                group_devices = {k: v for k, v in devices_data.items() if start_id <= k <= end_id}

                if group_devices:
                    output_file = docgen.generate_group_summary(group, group_devices)
                    print(f"  Group {group}: {output_file}")

            # Device profiles
            for device_id, device_data in devices_data.items():
                output_file = docgen.generate_device_profile(device_id, device_data)
                print(f"  Device 0x{device_id:04X}: {output_file}")

            print("\nDocumentation generation complete!")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
