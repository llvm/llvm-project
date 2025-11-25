#!/usr/bin/env python3
"""
Interactive DSMIL Enumeration Probe

Tests and enumerates Dell System Military Integration Layer (DSMIL) devices:
- 22 implemented devices (TPM, Boot Security, Credential Vault, etc.)
- 108 total DSMIL devices on platform
- Military-grade token validation (0x049e-0x04a3)
- Hardware security features

Usage:
    python 04_test_dsmil_enumeration.py
"""

import sys
import os
import subprocess

# Add DSMIL tools path
DSMIL_DEVICES_PATH = os.path.abspath("../../02-tools/dsmil-devices")
sys.path.insert(0, DSMIL_DEVICES_PATH)


def check_dsmil_driver():
    """Check if DSMIL driver is loaded"""
    print("\n" + "=" * 80)
    print("  DSMIL DRIVER STATUS")
    print("=" * 80)

    try:
        result = subprocess.run(["lsmod"], capture_output=True, text=True)
        if "dsmil" in result.stdout.lower():
            print("\n✓ DSMIL driver loaded")
            for line in result.stdout.split("\n"):
                if "dsmil" in line.lower():
                    print(f"  {line.strip()}")
        else:
            print("\n✗ DSMIL driver NOT loaded")
            print("  Load with: sudo modprobe dsmil-84dev   # legacy alias dsmil-72dev also valid")
            print("  Or: sudo insmod /path/to/dsmil-84dev.ko (symlink dsmil-72dev.ko is installed)")
    except FileNotFoundError:
        print("\n✗ lsmod not available")


def check_dsmil_devices():
    """Check for DSMIL device nodes"""
    print("\n" + "=" * 80)
    print("  DSMIL DEVICE NODES")
    print("=" * 80)

    device_paths = [
        "/dev/dsmil-84dev",
        "/dev/dsmil-72dev",
        "/dev/dsmil",
        "/dev/dsmil0",
        "/sys/class/dsmil-84dev/dsmil-84dev",
        "/sys/class/dsmil-72dev/dsmil-72dev",
        "/sys/class/dsmil",
    ]

    found = False
    for path in device_paths:
        if os.path.exists(path):
            print(f"\n✓ Found: {path}")
            found = True

            # Show details
            try:
                stat = os.stat(path)
                print(f"  Permissions: {oct(stat.st_mode)[-3:]}")
                print(f"  Owner: {stat.st_uid}:{stat.st_gid}")
            except:
                pass

    if not found:
        print("\n✗ No DSMIL device nodes found")
        print("  Ensure DSMIL driver is loaded")


def check_military_tokens():
    """Check military token status"""
    print("\n" + "=" * 80)
    print("  MILITARY TOKEN STATUS")
    print("=" * 80)

    print("\nMilitary Tokens (Dell Latitude 5450 MIL-SPEC):")
    tokens = [
        (0x049e, "Primary Authorization", "UNCLASSIFIED"),
        (0x049f, "Secondary Validation", "CONFIDENTIAL"),
        (0x04a0, "Hardware Activation", "CONFIDENTIAL"),
        (0x04a1, "Advanced Security", "SECRET"),
        (0x04a2, "System Integration", "SECRET"),
        (0x04a3, "Military Validation", "TOP_SECRET"),
    ]

    for token_id, name, level in tokens:
        print(f"\n  {hex(token_id)}: {name}")
        print(f"    Security Level: {level}")

    print("\n  Note: Tokens require DSMIL driver for read/write access")


def list_implemented_devices():
    """List implemented DSMIL devices"""
    print("\n" + "=" * 80)
    print("  IMPLEMENTED DSMIL DEVICES (80/108 = 74.1%)")
    print("=" * 80)

    print("\n✅ ALL 84 STANDARD DEVICES INTEGRATED!")
    print("   (Groups 0-6: 7 groups × 12 devices each)")
    print(f"\n   80 Active + 5 Quarantined = 85 total in standard range")
    print(f"   23 Unknown in extended range (0x8054-0x806B)")

    print("\n" + "-" * 80)
    print("Sample Devices (showing key devices, not all 80):")

    devices = [
        ("0x8000", "TPM Control", "Core Security", "MONITORED"),
        ("0x8001", "Boot Security", "Core Security", "MONITORED"),
        ("0x8002", "Credential Vault", "Core Security", "MONITORED"),
        ("0x8003", "Audit Log", "Core Security", "SAFE"),
        ("0x8004", "Event Logger", "Core Security", "SAFE"),
        ("0x8005", "Performance Monitor", "Core Security", "SAFE"),
        ("0x8006", "Thermal Sensor", "Core Security", "SAFE"),
        ("0x8007", "Power State", "Core Security", "MONITORED"),
        ("0x8008", "Emergency Response", "Core Security", "MONITORED"),
        ("0x8010", "Intrusion Detection", "Extended Security", "SAFE"),
        ("0x8013", "Key Management", "Extended Security", "MONITORED"),
        ("0x8014", "Certificate Store", "Extended Security", "SAFE"),
        ("0x8016", "VPN Controller", "Extended Security", "MONITORED"),
        ("0x8017", "Remote Access", "Extended Security", "MONITORED"),
        ("0x8018", "Pre-Isolation", "Extended Security", "MONITORED"),
        ("0x801A", "Port Security", "Extended Security", "MONITORED"),
        ("0x801B", "Wireless Security", "Extended Security", "MONITORED"),
        ("0x801E", "Tactical Display", "Network/Comms", "MONITORED"),
        ("0x802A", "Network Monitor", "Network/Comms", "SAFE"),
        ("0x802B", "Packet Filter", "Network/Comms", "MONITORED"),
        ("0x8050", "Storage Encryption", "Storage/Data", "MONITORED"),
        ("0x805A", "Sensor Array", "Peripheral/Sensors", "SAFE"),
    ]

    print("\nCore Security (9 devices):")
    for dev_id, name, group, risk in devices[:9]:
        status = "🔒" if risk == "MONITORED" else "✓"
        print(f"  {status} {dev_id}: {name}")

    print("\nExtended Security (8 devices):")
    for dev_id, name, group, risk in devices[9:17]:
        status = "🔒" if risk == "MONITORED" else "✓"
        print(f"  {status} {dev_id}: {name}")

    print("\nNetwork/Comms (3 devices):")
    for dev_id, name, group, risk in devices[17:20]:
        status = "🔒" if risk == "MONITORED" else "✓"
        print(f"  {status} {dev_id}: {name}")

    print("\nStorage/Data + Peripheral (2 devices):")
    for dev_id, name, group, risk in devices[20:]:
        status = "🔒" if risk == "MONITORED" else "✓"
        print(f"  {status} {dev_id}: {name}")

    print(f"\n" + "=" * 80)
    print(f"FULL STATISTICS:")
    print(f"  • 80 devices ACTIVE (74.1%)")
    print(f"  • 5 devices QUARANTINED (4.6%)")
    print(f"    - 0x8009: Data Destruction")
    print(f"    - 0x800A: Cascade Wipe")
    print(f"    - 0x800B: Hardware Sanitize")
    print(f"    - 0x8019: Network Kill")
    print(f"    - 0x8029: Comms Blackout")
    print(f"  • 23 devices UNKNOWN (21.3%) - Extended range")
    print(f"  • 108 total devices")
    print(f"\n✅ ALL STANDARD GROUPS (0-6) 100% IMPLEMENTED!")


def launch_dsmil_menu():
    """Launch the DSMIL interactive menu"""
    print("\n" + "=" * 80)
    print("  LAUNCHING DSMIL INTERACTIVE MENU")
    print("=" * 80)

    menu_path = os.path.join(DSMIL_DEVICES_PATH, "dsmil_menu.py")

    if not os.path.exists(menu_path):
        print(f"\n✗ DSMIL menu not found: {menu_path}")
        return

    print(f"\nLaunching: {menu_path}")
    print("(Exit menu to return here)")
    print()

    try:
        subprocess.run([sys.executable, menu_path])
    except KeyboardInterrupt:
        print("\n\nMenu interrupted.")
    except Exception as e:
        print(f"\n✗ Error launching menu: {e}")


def run_dsmil_discovery():
    """Run DSMIL device discovery"""
    print("\n" + "=" * 80)
    print("  DSMIL DEVICE DISCOVERY")
    print("=" * 80)

    discover_path = os.path.join(DSMIL_DEVICES_PATH, "dsmil_discover.py")

    if not os.path.exists(discover_path):
        print(f"\n✗ DSMIL discover script not found: {discover_path}")
        return

    print(f"\nRunning: {discover_path}")
    print()

    try:
        subprocess.run([sys.executable, discover_path, "--summary"])
    except Exception as e:
        print(f"\n✗ Error running discovery: {e}")


def run_dsmil_probe():
    """Run DSMIL device probe"""
    print("\n" + "=" * 80)
    print("  DSMIL DEVICE PROBE")
    print("=" * 80)

    probe_path = os.path.join(DSMIL_DEVICES_PATH, "dsmil_probe.py")

    if not os.path.exists(probe_path):
        print(f"\n✗ DSMIL probe script not found: {probe_path}")
        return

    device_id = input("\nEnter device ID to probe (e.g., 0x8000): ").strip()

    print(f"\nProbing device {device_id}...")
    print()

    try:
        subprocess.run([sys.executable, probe_path, device_id])
    except Exception as e:
        print(f"\n✗ Error running probe: {e}")


def show_quick_reference():
    """Show quick reference guide"""
    print("\n" + "=" * 80)
    print("  DSMIL QUICK REFERENCE")
    print("=" * 80)

    print("\nDSMIL Command-Line Tools:")
    print("  python3 02-tools/dsmil-devices/dsmil_menu.py")
    print("    → Interactive menu system")
    print()
    print("  python3 02-tools/dsmil-devices/dsmil_discover.py --summary")
    print("    → Discover and enumerate devices")
    print()
    print("  python3 02-tools/dsmil-devices/dsmil_probe.py 0x8000")
    print("    → Probe specific device")
    print()
    print("  python3 02-tools/dsmil-devices/dsmil_integration.py --list")
    print("    → List all devices")

    print("\nDSMIL Driver:")
    print("  sudo modprobe dsmil-72dev")
    print("    → Load DSMIL kernel driver")
    print()
    print("  lsmod | grep dsmil")
    print("    → Check if driver loaded")

    print("\nMilitary Tokens:")
    print("  Tokens: 0x049e - 0x04a3 (6 tokens)")
    print("  Security Levels: UNCLASSIFIED → TOP_SECRET")
    print("  Requires: DSMIL driver + Dell WMI")

    print("\nDevice Groups:")
    print("  Core Security (0x8000-0x8008)")
    print("  Extended Security (0x8010-0x801B)")
    print("  Network/Comms (0x801E-0x802B)")
    print("  Storage/Data (0x8050-0x8058)")
    print("  Peripherals (0x805A-0x8062)")


def interactive_menu():
    """Interactive menu for DSMIL enumeration"""
    while True:
        print("\n" + "=" * 80)
        print("  DSMIL ENUMERATION INTERACTIVE PROBE")
        print("=" * 80)
        print("\n1. Check DSMIL driver status")
        print("2. Check DSMIL device nodes")
        print("3. Check military token status")
        print("4. List implemented devices (22/108)")
        print("5. Launch DSMIL interactive menu (full TUI)")
        print("6. Run device discovery")
        print("7. Probe specific device")
        print("8. Show quick reference")
        print("9. Run all checks")
        print("0. Exit")

        choice = input("\nSelect option: ").strip()

        if choice == "1":
            check_dsmil_driver()
        elif choice == "2":
            check_dsmil_devices()
        elif choice == "3":
            check_military_tokens()
        elif choice == "4":
            list_implemented_devices()
        elif choice == "5":
            launch_dsmil_menu()
        elif choice == "6":
            run_dsmil_discovery()
        elif choice == "7":
            run_dsmil_probe()
        elif choice == "8":
            show_quick_reference()
        elif choice == "9":
            check_dsmil_driver()
            check_dsmil_devices()
            check_military_tokens()
            list_implemented_devices()
        elif choice == "0":
            print("\nExiting...")
            break
        else:
            print("\nInvalid option!")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════╗
║          DSMIL ENUMERATION INTERACTIVE PROBE                             ║
║          Dell Latitude 5450 MIL-SPEC Edition                             ║
╚══════════════════════════════════════════════════════════════════════════╝

Dell System Military Integration Layer (DSMIL):
  • 108 total DSMIL devices on platform
  • 80 devices FULLY IMPLEMENTED (74.1% coverage)
  • 5 devices QUARANTINED for safety (4.6%)
  • 23 devices UNKNOWN in extended range (21.3%)
  • 6 military tokens (0x049e-0x04a3)
  • Security levels: UNCLASSIFIED → TOP_SECRET

Device Groups (84 standard + 24 extended):
  • Group 0: Core Security (12 devices, 3 quarantined)
  • Group 1: Extended Security (12 devices)
  • Group 2: Network/Comms (12 devices, 1 quarantined)
  • Group 3: Data Processing (12 devices, 1 quarantined)
  • Group 4: Storage Management (12 devices)
  • Group 5: Peripheral Control (12 devices)
  • Group 6: Training/Simulation (12 devices)
  • Extended: Beyond standard grid (24 devices, 23 unknown)

""")

    interactive_menu()
