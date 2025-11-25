#!/usr/bin/env python3
"""
Device Capability Extractor

Extracts all methods, registers, and capabilities from DSMIL device implementations.
Creates a comprehensive JSON catalog of device capabilities.

Usage:
    python3 extract-device-capabilities.py
    python3 extract-device-capabilities.py --output capabilities.json
    python3 extract-device-capabilities.py --verbose
"""

import os
import ast
import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


def extract_class_docstring(class_node: ast.ClassDef) -> str:
    """Extract docstring from class definition"""
    return ast.get_docstring(class_node) or "No description available"


def extract_method_signature(method: ast.FunctionDef) -> Dict[str, Any]:
    """Extract method signature details"""
    args = []
    for arg in method.args.args:
        if arg.arg != 'self':
            # Get type annotation if available
            annotation = None
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    annotation = arg.annotation.id
                elif isinstance(arg.annotation, ast.Constant):
                    annotation = str(arg.annotation.value)

            args.append({
                "name": arg.arg,
                "type": annotation
            })

    # Get return type
    return_type = None
    if method.returns:
        if isinstance(method.returns, ast.Name):
            return_type = method.returns.id

    return {
        "name": method.name,
        "args": args,
        "return_type": return_type,
        "docstring": ast.get_docstring(method)
    }


def extract_register_map(class_node: ast.ClassDef) -> Dict[str, Any]:
    """Extract register definitions from class"""
    registers = {}

    for item in class_node.body:
        # Look for register constants (REG_*)
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    if target.id.startswith('REG_'):
                        reg_name = target.id.replace('REG_', '')

                        # Get value
                        value = None
                        if isinstance(item.value, ast.Constant):
                            value = item.value.value

                        registers[reg_name] = {
                            "constant": target.id,
                            "offset": value
                        }

    return registers


def extract_constants(class_node: ast.ClassDef) -> Dict[str, List[str]]:
    """Extract constant definitions from class"""
    constants = defaultdict(list)

    for item in class_node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    name = target.id

                    # Categorize constants
                    if name.startswith('STATUS_'):
                        constants['status_bits'].append(name)
                    elif name.startswith('CAP_'):
                        constants['capabilities'].append(name)
                    elif name.startswith('CMD_'):
                        constants['commands'].append(name)
                    elif name.startswith('ERROR_'):
                        constants['errors'].append(name)
                    elif name.startswith('MODE_'):
                        constants['modes'].append(name)

    return dict(constants)


def extract_device_info(file_path: Path) -> Dict[str, Any]:
    """Extract complete device information from Python file"""

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Find the main device class (inherits from DSMILDeviceBase)
        device_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it inherits from DSMILDeviceBase
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        if base.id == 'DSMILDeviceBase':
                            device_class = node
                            break

        if not device_class:
            return None

        # Extract device metadata from filename
        filename = file_path.stem  # e.g., "device_0x8000_tpm_control"
        parts = filename.split('_')

        device_info = {
            "device_id": parts[1] if len(parts) > 1 else "unknown",
            "name": device_class.name,
            "file": str(file_path.name),
            "description": extract_class_docstring(device_class),
            "public_methods": [],
            "private_methods": [],
            "registers": {},
            "constants": {},
            "total_operations": 0
        }

        # Extract methods
        for item in device_class.body:
            if isinstance(item, ast.FunctionDef):
                method_info = extract_method_signature(item)

                if item.name.startswith('_') and item.name != '__init__':
                    device_info['private_methods'].append(method_info)
                elif item.name != '__init__':
                    device_info['public_methods'].append(method_info)

        # Extract registers
        device_info['registers'] = extract_register_map(device_class)

        # Extract constants
        device_info['constants'] = extract_constants(device_class)

        # Count total operations
        device_info['total_operations'] = len(device_info['public_methods'])

        return device_info

    except Exception as e:
        print(f"Error processing {file_path}: {e}", file=sys.stderr)
        return None


def generate_summary(all_devices: Dict[str, Any]) -> Dict[str, Any]:
    """Generate summary statistics"""

    total_devices = len(all_devices)
    total_operations = sum(dev['total_operations'] for dev in all_devices.values())
    total_registers = sum(len(dev['registers']) for dev in all_devices.values())

    # Find most complex devices
    by_operations = sorted(
        all_devices.items(),
        key=lambda x: x[1]['total_operations'],
        reverse=True
    )

    # Count by device group (based on device_id range)
    by_group = defaultdict(list)
    for dev_id, dev_info in all_devices.items():
        try:
            device_num = int(dev_id, 16)
            if 0x8000 <= device_num <= 0x800B:
                group = "Group 0: Core Security"
            elif 0x800C <= device_num <= 0x8017:
                group = "Group 1: Extended Security"
            elif 0x8018 <= device_num <= 0x8023:
                group = "Group 2: Network/Comms"
            elif 0x8024 <= device_num <= 0x802F:
                group = "Group 3: Data Processing"
            elif 0x8030 <= device_num <= 0x803B:
                group = "Group 4: Storage Management"
            elif 0x803C <= device_num <= 0x8047:
                group = "Group 5: Peripheral Control"
            elif 0x8048 <= device_num <= 0x8053:
                group = "Group 6: Training/Simulation"
            else:
                group = "Extended Range"

            by_group[group].append(dev_id)
        except:
            pass

    return {
        "total_devices": total_devices,
        "total_operations": total_operations,
        "total_registers": total_registers,
        "average_operations_per_device": round(total_operations / total_devices, 1) if total_devices > 0 else 0,
        "most_complex_devices": [
            {
                "device_id": dev_id,
                "name": dev_info['name'],
                "operations": dev_info['total_operations']
            }
            for dev_id, dev_info in by_operations[:10]
        ],
        "devices_by_group": {group: len(devices) for group, devices in by_group.items()}
    }


def main():
    parser = argparse.ArgumentParser(description="Extract DSMIL device capabilities")
    parser.add_argument("--output", "-o", default="DSMIL_DEVICE_CAPABILITIES.json",
                       help="Output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--device-dir", "-d",
                       default="02-tools/dsmil-devices/devices",
                       help="Device directory")

    args = parser.parse_args()

    device_dir = Path(args.device_dir)

    if not device_dir.exists():
        print(f"Error: Device directory not found: {device_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("  DSMIL Device Capability Extractor")
    print("=" * 80)
    print(f"\nScanning device directory: {device_dir}")

    # Scan all device files
    device_files = sorted(device_dir.glob("device_0x*.py"))
    print(f"Found {len(device_files)} device files")

    all_devices = {}
    processed = 0
    skipped = 0

    for device_file in device_files:
        if args.verbose:
            print(f"\nProcessing: {device_file.name}")

        device_info = extract_device_info(device_file)

        if device_info:
            all_devices[device_info['device_id']] = device_info
            processed += 1

            if args.verbose:
                print(f"  ✓ {device_info['name']}")
                print(f"    Operations: {device_info['total_operations']}")
                print(f"    Registers: {len(device_info['registers'])}")
        else:
            skipped += 1
            if args.verbose:
                print(f"  ✗ Skipped (not a device class)")

    print(f"\n" + "=" * 80)
    print(f"Results:")
    print(f"  Processed: {processed} devices")
    print(f"  Skipped: {skipped} files")

    # Generate summary
    summary = generate_summary(all_devices)

    print(f"\nSummary Statistics:")
    print(f"  Total operations: {summary['total_operations']}")
    print(f"  Total registers: {summary['total_registers']}")
    print(f"  Average operations per device: {summary['average_operations_per_device']}")

    print(f"\nDevices by group:")
    for group, count in sorted(summary['devices_by_group'].items()):
        print(f"  {group}: {count} devices")

    print(f"\nMost complex devices:")
    for dev in summary['most_complex_devices'][:5]:
        print(f"  {dev['device_id']}: {dev['name']} ({dev['operations']} operations)")

    # Save to JSON
    output = {
        "summary": summary,
        "devices": all_devices,
        "extraction_date": "2025-11-08",
        "total_devices": len(all_devices)
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")

    print(f"\n" + "=" * 80)
    print("Extraction complete!")
    print("\nNext steps:")
    print("  1. Review the JSON file for completeness")
    print("  2. Generate individual device documentation")
    print("  3. Create interactive capability browser")


if __name__ == "__main__":
    main()
