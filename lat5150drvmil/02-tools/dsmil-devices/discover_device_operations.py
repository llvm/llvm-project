#!/usr/bin/env python3
"""
Device Operation Discovery Tool
Discovers all available operations for each DSMIL device
to help populate device-specific TUI menus
"""

import sys
import os
import inspect
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from device_base import DSMILDeviceBase
from dsmil_auto_discover import discover_device_files, load_device_class

def get_device_operations(device_class):
    """Get all public methods of a device that aren't from DSMILDeviceBase"""
    # Get base class methods
    base_methods = set(dir(DSMILDeviceBase))

    # Get device methods
    device_methods = set(dir(device_class))

    # Find device-specific public methods
    specific_methods = []
    for method_name in sorted(device_methods - base_methods):
        # Skip private/magic methods
        if method_name.startswith('_'):
            continue

        attr = getattr(device_class, method_name)
        if callable(attr):
            # Get method signature if possible
            try:
                sig = inspect.signature(attr)
                specific_methods.append({
                    'name': method_name,
                    'signature': str(sig),
                    'doc': (attr.__doc__ or '').strip().split('\n')[0]
                })
            except:
                specific_methods.append({
                    'name': method_name,
                    'signature': '(?)',
                    'doc': ''
                })

    return specific_methods

def discover_all_operations():
    """Discover operations for all devices"""
    print("=" * 80)
    print("DSMIL DEVICE OPERATION DISCOVERY")
    print("=" * 80)
    print()

    device_files = discover_device_files()

    # Group devices by whether they have custom operations
    devices_with_operations = {}
    devices_generic = []

    for device_file in device_files:
        device_id = device_file['device_id']
        module_name = device_file['module_name']
        filename = device_file['filename']

        # Extract device name from filename
        name_part = filename.replace(f"device_0x{device_id:04x}_", "").replace(".py", "")
        name = name_part.replace("_", " ").title()

        try:
            device_class = load_device_class(module_name)
            if not device_class:
                print(f"⚠ Could not load class from module {module_name}")
                continue

            operations = get_device_operations(device_class)

            # Determine group
            device_num = device_id - 0x8000
            group_num = device_num // 12
            group_names = [
                "Core Security",
                "Extended Security",
                "Network & Communications",
                "Data Processing",
                "Storage Management",
                "Peripheral Control",
                "Training & Simulation"
            ]
            group = group_names[group_num] if group_num < len(group_names) else "Unknown"

            if operations:
                devices_with_operations[device_id] = {
                    'name': name,
                    'operations': operations,
                    'group': group
                }
            else:
                devices_generic.append({
                    'id': device_id,
                    'name': name,
                    'group': group
                })
        except Exception as e:
            print(f"⚠ Error loading device 0x{device_id:04X}: {e}")

    # Report devices with custom operations
    print(f"✅ Devices with Custom Operations: {len(devices_with_operations)}")
    print(f"📋 Devices needing Custom Operations: {len(devices_generic)}")
    print()

    # Show devices with operations
    if devices_with_operations:
        print("=" * 80)
        print("DEVICES WITH CUSTOM OPERATIONS")
        print("=" * 80)
        print()

        for device_id, info in sorted(devices_with_operations.items()):
            print(f"Device 0x{device_id:04X}: {info['name']}")
            print(f"  Group: {info['group']}")
            print(f"  Operations ({len(info['operations'])}):")

            for op in info['operations']:
                doc_str = f" - {op['doc']}" if op['doc'] else ""
                print(f"    • {op['name']}{op['signature']}{doc_str}")
            print()

    # Show devices without custom operations
    if devices_generic:
        print("=" * 80)
        print("DEVICES USING GENERIC OPERATIONS")
        print("=" * 80)
        print()
        print("These devices use the 5 standard operations:")
        print("  1. Get Device Status")
        print("  2. Get Capabilities")
        print("  3. Read All Registers")
        print("  4. Get Statistics")
        print("  5. Perform Self-Test")
        print()

        for device in sorted(devices_generic, key=lambda x: x['id']):
            print(f"  0x{device['id']:04X}: {device['name']:30} [{device['group']}]")
        print()

        print("To add custom operations for these devices:")
        print("1. Edit the device class file (e.g., device_network.py)")
        print("2. Add public methods for device-specific operations")
        print("3. Add a custom menu function in dsmil_menu.py")
        print("4. Update the device_operations() method to call your custom menu")

    # Generate template for adding operations
    print()
    print("=" * 80)
    print("TEMPLATE FOR ADDING CUSTOM OPERATIONS")
    print("=" * 80)
    print()
    print("# Example: Add to device class (e.g., device_network.py)")
    print("""
class SomeDevice(DeviceBase):
    # ... existing code ...

    def custom_operation(self, param):
        \"\"\"Perform a custom operation\"\"\"
        # Implementation here
        return {"status": "success", "result": "data"}

    def another_operation(self):
        \"\"\"Another custom operation\"\"\"
        return {"status": "success"}
""")

    print("# Example: Add to dsmil_menu.py")
    print("""
def some_device_operations(self, device):
    \"\"\"Custom operations menu for SomeDevice\"\"\"
    while True:
        self.clear_screen()
        self.print_header(f"Device Operations: {device.name}")

        print(f"  Device ID: 0x{device.device_id:04X}")
        print(f"  State: {device.state.value}\\n")

        print("  [1] Custom Operation")
        print("  [2] Another Operation")
        print("  [0] Back")

        choice = self.get_input("Select operation", ["0", "1", "2"])

        if choice == "0":
            break
        elif choice == "1":
            result = device.custom_operation("param")
            self.display_result(result)
        elif choice == "2":
            result = device.another_operation()
            self.display_result(result)
""")

    print("# Then update device_operations() in dsmil_menu.py:")
    print("""
elif device_id == 0x1234:  # Your device ID
    self.some_device_operations(device)
""")

    return {
        'with_operations': len(devices_with_operations),
        'generic': len(devices_generic),
        'total': len(devices_with_operations) + len(devices_generic)
    }

if __name__ == "__main__":
    try:
        stats = discover_all_operations()
        print()
        print("=" * 80)
        print(f"Summary: {stats['with_operations']}/{stats['total']} devices have custom operations")
        print(f"         {stats['generic']} devices use generic operations")
        print("=" * 80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
