#!/usr/bin/env python3
"""
DSMIL Command-Line Interface

Comprehensive CLI for DSMIL device enumeration, exploration, and management.
Designed for use with Claude Code and direct terminal interaction.

Usage:
    python3 dsmil-cli.py list                           # List all devices
    python3 dsmil-cli.py info 0x8000                    # Show device info
    python3 dsmil-cli.py operations 0x8000              # List device operations
    python3 dsmil-cli.py search "TPM"                   # Search devices by name
    python3 dsmil-cli.py stats                          # Show statistics
    python3 dsmil-cli.py groups                         # List devices by group
    python3 dsmil-cli.py complex --top 10               # Show most complex devices
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_capabilities(cap_file: str = "DSMIL_DEVICE_CAPABILITIES.json") -> Dict[str, Any]:
    """Load device capability catalog"""
    cap_path = Path(cap_file)
    if not cap_path.exists():
        print(f"Error: Capability file not found: {cap_file}", file=sys.stderr)
        print("Run: python3 scripts/extract-device-capabilities.py", file=sys.stderr)
        sys.exit(1)

    with open(cap_path, 'r') as f:
        return json.load(f)


def get_risk_level(device_id: str) -> tuple:
    """Get risk level and color for device"""
    try:
        device_num = int(device_id, 16)
        if device_num in [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]:
            return ("QUARANTINED", "🔴")
        elif device_num in [0x8000, 0x8001, 0x8002, 0x8007, 0x8008]:
            return ("MONITORED", "🟡")
        else:
            return ("SAFE", "🟢")
    except:
        return ("UNKNOWN", "⚪")


def get_group_name(device_id: str) -> str:
    """Get group name for device"""
    try:
        device_num = int(device_id, 16)
        if 0x8000 <= device_num <= 0x800B:
            return "Group 0: Core Security"
        elif 0x800C <= device_num <= 0x8017:
            return "Group 1: Extended Security"
        elif 0x8018 <= device_num <= 0x8023:
            return "Group 2: Network/Comms"
        elif 0x8024 <= device_num <= 0x802F:
            return "Group 3: Data Processing"
        elif 0x8030 <= device_num <= 0x803B:
            return "Group 4: Storage"
        elif 0x803C <= device_num <= 0x8047:
            return "Group 5: Peripherals"
        elif 0x8048 <= device_num <= 0x8053:
            return "Group 6: Training"
        else:
            return "Extended Range"
    except:
        return "Unknown"


def cmd_list(data: Dict[str, Any], args):
    """List all devices"""
    devices = data['devices']

    print("\n" + "=" * 100)
    print("  DSMIL DEVICE LIST ({} devices)".format(len(devices)))
    print("=" * 100)

    # Header
    print(f"\n{'ID':<10} {'Risk':<4} {'Name':<35} {'Ops':<5} {'Regs':<5} {'Group':<20}")
    print("-" * 100)

    # Sort devices by ID
    for dev_id in sorted(devices.keys(), key=lambda x: int(x, 16)):
        dev_info = devices[dev_id]
        risk_level, risk_icon = get_risk_level(dev_id)
        group = get_group_name(dev_id)

        # Truncate long names
        name = dev_info['name']
        if len(name) > 33:
            name = name[:30] + "..."

        print(f"{dev_id:<10} {risk_icon:<4} {name:<35} {dev_info['total_operations']:<5} "
              f"{len(dev_info['registers']):<5} {group:<20}")

    print("\n" + "=" * 100)
    print(f"Total: {len(devices)} devices | 🟢 SAFE  🟡 MONITORED  🔴 QUARANTINED")


def cmd_info(data: Dict[str, Any], args):
    """Show detailed device information"""
    devices = data['devices']
    dev_id = args.device_id.lower()

    if dev_id not in devices:
        print(f"Error: Device {dev_id} not found", file=sys.stderr)
        return 1

    dev_info = devices[dev_id]
    risk_level, risk_icon = get_risk_level(dev_id)
    group = get_group_name(dev_id)

    print("\n" + "=" * 80)
    print(f"  DEVICE {dev_id}: {dev_info['name']}")
    print("=" * 80)

    print(f"\n📋 Device Information:")
    print(f"  ID: {dev_id} ({int(dev_id, 16)})")
    print(f"  Name: {dev_info['name']}")
    print(f"  Group: {group}")
    print(f"  Risk Level: {risk_icon} {risk_level}")
    print(f"  File: {dev_info['file']}")

    print(f"\n📊 Statistics:")
    print(f"  Operations: {dev_info['total_operations']}")
    print(f"  Registers: {len(dev_info['registers'])}")
    print(f"  Private Methods: {len(dev_info['private_methods'])}")

    # Description
    if dev_info.get('description'):
        desc = dev_info['description'].strip()
        if desc and desc != "No description available":
            print(f"\n📝 Description:")
            # Print first 3 lines
            lines = desc.split('\n')[:3]
            for line in lines:
                print(f"  {line}")
            if len(desc.split('\n')) > 3:
                print("  ...")

    # Show first few operations
    if dev_info['public_methods']:
        print(f"\n⚙️  Operations (showing first 10):")
        for i, op in enumerate(dev_info['public_methods'][:10], 1):
            args = ", ".join([arg["name"] for arg in op.get("args", [])])
            print(f"  {i:2}. {op['name']}({args})")
        if len(dev_info['public_methods']) > 10:
            print(f"  ... and {len(dev_info['public_methods']) - 10} more")

    # Show registers
    if dev_info['registers']:
        print(f"\n🔧 Registers:")
        for reg_name, reg_info in list(dev_info['registers'].items())[:5]:
            offset = f"0x{reg_info['offset']:02X}" if reg_info.get('offset') is not None else "N/A"
            print(f"  - {reg_name}: {offset}")
        if len(dev_info['registers']) > 5:
            print(f"  ... and {len(dev_info['registers']) - 5} more")

    print("\n" + "=" * 80)


def cmd_operations(data: Dict[str, Any], args):
    """List all operations for a device"""
    devices = data['devices']
    dev_id = args.device_id.lower()

    if dev_id not in devices:
        print(f"Error: Device {dev_id} not found", file=sys.stderr)
        return 1

    dev_info = devices[dev_id]

    print("\n" + "=" * 80)
    print(f"  OPERATIONS: {dev_id} - {dev_info['name']}")
    print("=" * 80)
    print(f"\nTotal Operations: {dev_info['total_operations']}\n")

    for i, op in enumerate(dev_info['public_methods'], 1):
        args_list = ", ".join([arg["name"] for arg in op.get("args", [])])
        print(f"{i:3}. {op['name']}({args_list})")

        # Show docstring if requested
        if args.verbose and op.get('docstring'):
            doc = op['docstring'].strip()
            if doc:
                # Print first line of docstring
                first_line = doc.split('\n')[0]
                print(f"     {first_line}")
        print()


def cmd_search(data: Dict[str, Any], args):
    """Search devices by name or description"""
    devices = data['devices']
    query = args.query.lower()

    matches = []
    for dev_id, dev_info in devices.items():
        if (query in dev_info['name'].lower() or
            query in dev_info.get('description', '').lower() or
            query in dev_id.lower()):
            matches.append((dev_id, dev_info))

    print("\n" + "=" * 80)
    print(f"  SEARCH RESULTS: '{args.query}' ({len(matches)} matches)")
    print("=" * 80 + "\n")

    if not matches:
        print("No devices found matching query.")
        return

    for dev_id, dev_info in sorted(matches, key=lambda x: int(x[0], 16)):
        risk_level, risk_icon = get_risk_level(dev_id)
        print(f"{risk_icon} {dev_id}: {dev_info['name']}")
        print(f"   Operations: {dev_info['total_operations']}, Group: {get_group_name(dev_id)}")
        print()


def cmd_stats(data: Dict[str, Any], args):
    """Show comprehensive statistics"""
    summary = data['summary']

    print("\n" + "=" * 80)
    print("  DSMIL DEVICE STATISTICS")
    print("=" * 80)

    print(f"\n📊 Overall Statistics:")
    print(f"  Total Devices: {summary['total_devices']}")
    print(f"  Total Operations: {summary['total_operations']}")
    print(f"  Total Registers: {summary['total_registers']}")
    print(f"  Average Operations per Device: {summary['average_operations_per_device']}")

    print(f"\n📦 Devices by Group:")
    for group, count in sorted(summary['devices_by_group'].items()):
        print(f"  {group}: {count} devices")

    print(f"\n🏆 Most Complex Devices:")
    for i, dev in enumerate(summary['most_complex_devices'][:5], 1):
        print(f"  {i}. {dev['device_id']}: {dev['name']} ({dev['operations']} operations)")

    # Risk level breakdown
    devices = data['devices']
    safe = sum(1 for d in devices.keys() if get_risk_level(d)[0] == "SAFE")
    monitored = sum(1 for d in devices.keys() if get_risk_level(d)[0] == "MONITORED")
    quarantined = sum(1 for d in devices.keys() if get_risk_level(d)[0] == "QUARANTINED")

    print(f"\n🔒 Risk Level Breakdown:")
    print(f"  🟢 SAFE: {safe} devices")
    print(f"  🟡 MONITORED: {monitored} devices")
    print(f"  🔴 QUARANTINED: {quarantined} devices")

    print("\n" + "=" * 80)


def cmd_groups(data: Dict[str, Any], args):
    """List devices organized by group"""
    devices = data['devices']

    # Organize by group
    groups = {}
    for dev_id, dev_info in devices.items():
        group = get_group_name(dev_id)
        if group not in groups:
            groups[group] = []
        groups[group].append((dev_id, dev_info))

    print("\n" + "=" * 80)
    print("  DEVICES BY GROUP")
    print("=" * 80)

    for group in sorted(groups.keys()):
        group_devices = sorted(groups[group], key=lambda x: int(x[0], 16))
        print(f"\n{group} ({len(group_devices)} devices):")
        print("-" * 80)

        for dev_id, dev_info in group_devices:
            risk_level, risk_icon = get_risk_level(dev_id)
            print(f"  {risk_icon} {dev_id}: {dev_info['name']:<40} ({dev_info['total_operations']} ops)")

    print("\n" + "=" * 80)


def cmd_complex(data: Dict[str, Any], args):
    """Show most complex devices"""
    devices = data['devices']
    top_n = args.top

    # Sort by operation count
    sorted_devices = sorted(
        devices.items(),
        key=lambda x: x[1]['total_operations'],
        reverse=True
    )[:top_n]

    print("\n" + "=" * 80)
    print(f"  TOP {top_n} MOST COMPLEX DEVICES")
    print("=" * 80 + "\n")

    for i, (dev_id, dev_info) in enumerate(sorted_devices, 1):
        risk_level, risk_icon = get_risk_level(dev_id)
        group = get_group_name(dev_id)

        print(f"{i:2}. {risk_icon} {dev_id}: {dev_info['name']}")
        print(f"    Operations: {dev_info['total_operations']}, Registers: {len(dev_info['registers'])}, Group: {group}")
        print()


def cmd_export(data: Dict[str, Any], args):
    """Export device list to various formats"""
    devices = data['devices']

    if args.format == 'csv':
        print("device_id,name,operations,registers,risk_level,group")
        for dev_id in sorted(devices.keys(), key=lambda x: int(x, 16)):
            dev_info = devices[dev_id]
            risk_level, _ = get_risk_level(dev_id)
            group = get_group_name(dev_id)
            print(f"{dev_id},{dev_info['name']},{dev_info['total_operations']},"
                  f"{len(dev_info['registers'])},{risk_level},{group}")

    elif args.format == 'json':
        # Export simplified format
        export_data = []
        for dev_id, dev_info in devices.items():
            export_data.append({
                "id": dev_id,
                "name": dev_info['name'],
                "operations": dev_info['total_operations'],
                "registers": len(dev_info['registers']),
                "risk": get_risk_level(dev_id)[0],
                "group": get_group_name(dev_id)
            })
        print(json.dumps(export_data, indent=2))

    elif args.format == 'markdown':
        print("# DSMIL Device List\n")
        print("| ID | Risk | Name | Operations | Registers | Group |")
        print("|---|---|---|---|---|---|")
        for dev_id in sorted(devices.keys(), key=lambda x: int(x, 16)):
            dev_info = devices[dev_id]
            risk_level, risk_icon = get_risk_level(dev_id)
            group = get_group_name(dev_id)
            print(f"| {dev_id} | {risk_icon} | {dev_info['name']} | {dev_info['total_operations']} | "
                  f"{len(dev_info['registers'])} | {group} |")


def main():
    parser = argparse.ArgumentParser(
        description="DSMIL CLI - Command-line interface for DSMIL device enumeration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    # List all devices
  %(prog)s info 0x8000             # Show detailed device info
  %(prog)s operations 0x8000 -v    # List operations with descriptions
  %(prog)s search "TPM"            # Search for devices
  %(prog)s stats                   # Show statistics
  %(prog)s groups                  # List by group
  %(prog)s complex --top 5         # Show top 5 complex devices
  %(prog)s export --format csv     # Export to CSV
        """
    )

    parser.add_argument("--cap-file", "-c",
                       default="DSMIL_DEVICE_CAPABILITIES.json",
                       help="Capability JSON file")

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    parser_list = subparsers.add_parser('list', help='List all devices')

    # Info command
    parser_info = subparsers.add_parser('info', help='Show device information')
    parser_info.add_argument('device_id', help='Device ID (e.g., 0x8000)')

    # Operations command
    parser_ops = subparsers.add_parser('operations', help='List device operations')
    parser_ops.add_argument('device_id', help='Device ID')
    parser_ops.add_argument('-v', '--verbose', action='store_true',
                           help='Show operation descriptions')

    # Search command
    parser_search = subparsers.add_parser('search', help='Search devices')
    parser_search.add_argument('query', help='Search query')

    # Stats command
    parser_stats = subparsers.add_parser('stats', help='Show statistics')

    # Groups command
    parser_groups = subparsers.add_parser('groups', help='List devices by group')

    # Complex command
    parser_complex = subparsers.add_parser('complex', help='Show most complex devices')
    parser_complex.add_argument('--top', '-n', type=int, default=10,
                               help='Number of devices to show')

    # Export command
    parser_export = subparsers.add_parser('export', help='Export device list')
    parser_export.add_argument('--format', '-f', choices=['csv', 'json', 'markdown'],
                              default='csv', help='Export format')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Load capability data
    data = load_capabilities(args.cap_file)

    # Execute command
    commands = {
        'list': cmd_list,
        'info': cmd_info,
        'operations': cmd_operations,
        'search': cmd_search,
        'stats': cmd_stats,
        'groups': cmd_groups,
        'complex': cmd_complex,
        'export': cmd_export,
    }

    if args.command in commands:
        return commands[args.command](data, args) or 0
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
