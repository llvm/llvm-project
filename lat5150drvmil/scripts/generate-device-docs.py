#!/usr/bin/env python3
"""
Device Documentation Generator

Generates individual markdown documentation files for each DSMIL device
based on the extracted capability catalog.

Usage:
    python3 generate-device-docs.py
    python3 generate-device-docs.py --input capabilities.json --output-dir docs/devices
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def format_method_signature(method: Dict[str, Any]) -> str:
    """Format method signature for documentation"""
    args = ", ".join([arg["name"] for arg in method.get("args", [])])
    return f"`{method['name']}({args})`"


def format_device_id_range(device_id: str) -> str:
    """Get the device group and range description"""
    try:
        device_num = int(device_id, 16)
        if 0x8000 <= device_num <= 0x800B:
            return "Group 0: Core Security (0x8000-0x800B)"
        elif 0x800C <= device_num <= 0x8017:
            return "Group 1: Extended Security (0x800C-0x8017)"
        elif 0x8018 <= device_num <= 0x8023:
            return "Group 2: Network/Communications (0x8018-0x8023)"
        elif 0x8024 <= device_num <= 0x802F:
            return "Group 3: Data Processing (0x8024-0x802F)"
        elif 0x8030 <= device_num <= 0x803B:
            return "Group 4: Storage Management (0x8030-0x803B)"
        elif 0x803C <= device_num <= 0x8047:
            return "Group 5: Peripheral Control (0x803C-0x8047)"
        elif 0x8048 <= device_num <= 0x8053:
            return "Group 6: Training/Simulation (0x8048-0x8053)"
        else:
            return "Extended Range (0x8054-0x806B)"
    except:
        return "Unknown Group"


def get_risk_level(device_id: str) -> str:
    """Determine risk level based on device ID and type"""
    try:
        device_num = int(device_id, 16)
        # Quarantined devices
        if device_num in [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]:
            return "🔴 QUARANTINED (Never Access)"
        # Security-critical devices
        elif device_num in [0x8000, 0x8001, 0x8002, 0x8007, 0x8008]:
            return "🟡 MONITORED (85% safe for READ)"
        else:
            return "🟢 SAFE"
    except:
        return "⚪ UNKNOWN"


def generate_device_doc(device_id: str, device_info: Dict[str, Any], output_dir: Path):
    """Generate markdown documentation for a single device"""

    doc_path = output_dir / f"{device_id}.md"

    with open(doc_path, 'w') as f:
        # Header
        f.write(f"# Device {device_id}: {device_info['name']}\n\n")

        # Metadata table
        f.write("## Device Information\n\n")
        f.write("| Property | Value |\n")
        f.write("|----------|-------|\n")
        f.write(f"| **Device ID** | `{device_id}` ({int(device_id, 16)}) |\n")
        f.write(f"| **Name** | {device_info['name']} |\n")
        f.write(f"| **Group** | {format_device_id_range(device_id)} |\n")
        f.write(f"| **Risk Level** | {get_risk_level(device_id)} |\n")
        f.write(f"| **Total Operations** | {device_info['total_operations']} |\n")
        f.write(f"| **Total Registers** | {len(device_info['registers'])} |\n")
        f.write(f"| **Implementation File** | `{device_info['file']}` |\n\n")

        # Description
        if device_info.get('description'):
            f.write("## Description\n\n")
            description = device_info['description'].strip()
            if description:
                f.write(f"{description}\n\n")
            else:
                f.write("*No description available*\n\n")

        # Operations
        if device_info['public_methods']:
            f.write(f"## Operations ({len(device_info['public_methods'])})\n\n")

            # Categorize operations
            core_ops = []
            config_ops = []
            advanced_ops = []

            for op in device_info['public_methods']:
                name = op['name']
                if name in ['initialize', 'get_status', 'get_capabilities']:
                    core_ops.append(op)
                elif name.startswith('set_') or name.startswith('configure_') or 'config' in name.lower():
                    config_ops.append(op)
                else:
                    advanced_ops.append(op)

            # Core operations
            if core_ops:
                f.write("### Core Operations\n\n")
                for op in core_ops:
                    f.write(f"#### {format_method_signature(op)}\n\n")
                    if op.get('docstring'):
                        f.write(f"{op['docstring']}\n\n")

                    # Parameters
                    if op.get('args'):
                        f.write("**Parameters:**\n")
                        for arg in op['args']:
                            arg_type = f" (`{arg['type']}`)" if arg.get('type') else ""
                            f.write(f"- `{arg['name']}`{arg_type}\n")
                        f.write("\n")

                    # Return type
                    if op.get('return_type'):
                        f.write(f"**Returns:** `{op['return_type']}`\n\n")

            # Configuration operations
            if config_ops:
                f.write("### Configuration Operations\n\n")
                for op in config_ops:
                    f.write(f"#### {format_method_signature(op)}\n\n")
                    if op.get('docstring'):
                        f.write(f"{op['docstring']}\n\n")

            # Advanced operations
            if advanced_ops:
                f.write("### Advanced Operations\n\n")
                for op in advanced_ops:
                    f.write(f"#### {format_method_signature(op)}\n\n")
                    if op.get('docstring'):
                        f.write(f"{op['docstring']}\n\n")

        # Registers
        if device_info['registers']:
            f.write(f"## Hardware Registers ({len(device_info['registers'])})\n\n")
            f.write("| Register Name | Constant | Offset |\n")
            f.write("|---------------|----------|--------|\n")
            for reg_name, reg_info in sorted(device_info['registers'].items()):
                offset = f"0x{reg_info['offset']:02X}" if reg_info.get('offset') is not None else "N/A"
                f.write(f"| {reg_name} | `{reg_info['constant']}` | {offset} |\n")
            f.write("\n")

        # Constants
        if device_info['constants']:
            f.write("## Device Constants\n\n")
            for const_type, const_list in device_info['constants'].items():
                if const_list:
                    # Format the type name nicely
                    type_name = const_type.replace('_', ' ').title()
                    f.write(f"### {type_name}\n\n")
                    for const in const_list:
                        f.write(f"- `{const}`\n")
                    f.write("\n")

        # Usage examples
        f.write("## Usage Example\n\n")
        f.write("```python\n")
        f.write("from dsmil_auto_discover import discover_all_devices\n\n")
        f.write("# Auto-discover all devices\n")
        f.write("registry = discover_all_devices()\n\n")
        f.write(f"# Get device {device_id}\n")
        f.write(f"device = registry.get_device({int(device_id, 16)})\n\n")
        f.write("# Initialize device\n")
        f.write("result = device.initialize()\n")
        f.write("if result.success:\n")
        f.write("    print(f'Device initialized: {device.name}')\n\n")
        f.write("# Get device status\n")
        f.write("status = device.get_status()\n")
        f.write("print(f'Device status: {status}')\n")
        f.write("```\n\n")

        # Safety notes for risky devices
        risk = get_risk_level(device_id)
        if "QUARANTINED" in risk:
            f.write("## ⚠️ CRITICAL SAFETY WARNING\n\n")
            f.write("**THIS DEVICE IS PERMANENTLY QUARANTINED**\n\n")
            f.write("This device performs destructive operations and is blocked at all system levels:\n")
            f.write("- Hardware access blocked\n")
            f.write("- Kernel driver blocks access\n")
            f.write("- Software registry blocks initialization\n")
            f.write("- Auto-discovery quarantine list\n\n")
            f.write("**NEVER attempt to access this device without explicit authorization.**\n\n")
        elif "MONITORED" in risk:
            f.write("## ⚠️ Safety Notes\n\n")
            f.write("This device is classified as MONITORED:\n")
            f.write("- READ operations are generally safe\n")
            f.write("- WRITE operations require careful review\n")
            f.write("- Always test in safe environment first\n")
            f.write("- Monitor for unexpected behavior\n\n")

        # Footer
        f.write("---\n\n")
        f.write("**Document Generated:** 2025-11-08  \n")
        f.write("**Framework Version:** 2.0.0 (Auto-Discovery)  \n")
        f.write("**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY\n")


def generate_index(devices: Dict[str, Any], output_dir: Path, summary: Dict[str, Any]):
    """Generate index/README for the device documentation"""

    index_path = output_dir / "README.md"

    with open(index_path, 'w') as f:
        f.write("# DSMIL Device Documentation\n\n")
        f.write("**Complete reference for all 80 implemented DSMIL devices**\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Devices:** {summary['total_devices']}\n")
        f.write(f"- **Total Operations:** {summary['total_operations']}\n")
        f.write(f"- **Total Registers:** {summary['total_registers']}\n")
        f.write(f"- **Average Operations per Device:** {summary['average_operations_per_device']}\n\n")

        # Most complex devices
        f.write("## Most Complex Devices\n\n")
        for dev in summary['most_complex_devices'][:10]:
            dev_id = dev['device_id']
            f.write(f"- [{dev_id}: {dev['name']}]({dev_id}.md) - {dev['operations']} operations\n")
        f.write("\n")

        # Devices by group
        f.write("## Devices by Group\n\n")

        groups = {
            "Group 0: Core Security": [],
            "Group 1: Extended Security": [],
            "Group 2: Network/Communications": [],
            "Group 3: Data Processing": [],
            "Group 4: Storage Management": [],
            "Group 5: Peripheral Control": [],
            "Group 6: Training/Simulation": [],
            "Extended Range": []
        }

        for dev_id, dev_info in devices.items():
            group = format_device_id_range(dev_id)
            # Extract just the group name
            for group_name in groups.keys():
                if group_name in group:
                    groups[group_name].append((dev_id, dev_info))
                    break

        for group_name, group_devices in groups.items():
            if group_devices:
                f.write(f"### {group_name} ({len(group_devices)} devices)\n\n")
                for dev_id, dev_info in sorted(group_devices):
                    risk = get_risk_level(dev_id)
                    risk_icon = risk.split()[0]
                    f.write(f"- {risk_icon} [{dev_id}: {dev_info['name']}]({dev_id}.md) - {dev_info['total_operations']} operations\n")
                f.write("\n")

        # Quick reference
        f.write("## Quick Reference\n\n")
        f.write("### Finding a Device\n\n")
        f.write("Click on any device ID above to see its complete documentation.\n\n")

        f.write("### Risk Levels\n\n")
        f.write("- 🟢 **SAFE** - Standard operations, low risk\n")
        f.write("- 🟡 **MONITORED** - Security-critical, READ operations safe\n")
        f.write("- 🔴 **QUARANTINED** - Destructive operations, permanently blocked\n\n")

        f.write("### Common Operations\n\n")
        f.write("All devices support these core operations:\n")
        f.write("- `initialize()` - Initialize the device\n")
        f.write("- `get_status()` - Get current device status\n")
        f.write("- `get_capabilities()` - Get device capabilities\n\n")

        # Footer
        f.write("---\n\n")
        f.write("**Generated:** 2025-11-08  \n")
        f.write("**Framework Version:** 2.0.0  \n")
        f.write("**Total Coverage:** 80/108 devices (74.1%)  \n")
        f.write("**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY\n")


def main():
    parser = argparse.ArgumentParser(description="Generate DSMIL device documentation")
    parser.add_argument("--input", "-i", default="DSMIL_DEVICE_CAPABILITIES.json",
                       help="Input JSON capability file")
    parser.add_argument("--output-dir", "-o", default="00-documentation/devices",
                       help="Output directory for documentation")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Load capability data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    with open(input_path, 'r') as f:
        data = json.load(f)

    devices = data['devices']
    summary = data['summary']

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  DSMIL Device Documentation Generator")
    print("=" * 80)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Devices: {len(devices)}")

    # Generate documentation for each device
    print(f"\nGenerating device documentation...")
    for i, (dev_id, dev_info) in enumerate(sorted(devices.items()), 1):
        if args.verbose:
            print(f"  [{i}/{len(devices)}] {dev_id}: {dev_info['name']}")
        generate_device_doc(dev_id, dev_info, output_dir)

    # Generate index
    print(f"\nGenerating index...")
    generate_index(devices, output_dir, summary)

    print(f"\n" + "=" * 80)
    print(f"Documentation generated successfully!")
    print(f"\nGenerated files:")
    print(f"  - {len(devices)} device documentation files")
    print(f"  - 1 index/README file")
    print(f"  - Total: {len(devices) + 1} markdown files")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Review generated documentation")
    print(f"  2. Create interactive capability browser")
    print(f"  3. Deploy to production documentation site")

    return 0


if __name__ == "__main__":
    exit(main())
