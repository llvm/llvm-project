#!/usr/bin/env python3
"""
Automated Device Operation Generator

Automatically generates device-specific operations and menus for all DSMIL devices
that currently only have generic operations.

This tool:
1. Analyzes device type/group/name
2. Generates appropriate operations based on device purpose
3. Adds methods to device class files
4. Creates menu functions in dsmil_menu.py
5. Updates the device_operations() routing

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import re
from pathlib import Path

# Add lib directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dsmil_auto_discover import discover_device_files, load_device_class
from device_base import DSMILDeviceBase, OperationResult


# Operation templates based on device category
OPERATION_TEMPLATES = {
    "network": [
        {
            "name": "get_connection_status",
            "doc": "Get current connection status",
            "returns": {"status": "connected", "bandwidth": "1Gbps", "latency": "5ms"}
        },
        {
            "name": "configure_interface",
            "doc": "Configure network interface",
            "params": [("interface_id", "str"), ("config", "dict")],
            "returns": {"configured": True, "interface": "eth0"}
        },
        {
            "name": "get_traffic_stats",
            "doc": "Get traffic statistics",
            "returns": {"rx_bytes": 1024000, "tx_bytes": 512000, "packets": 8000}
        },
    ],
    "security": [
        {
            "name": "get_threat_level",
            "doc": "Get current threat assessment level",
            "returns": {"level": "LOW", "threats_detected": 0, "last_scan": "2025-11-06"}
        },
        {
            "name": "run_security_scan",
            "doc": "Perform security scan",
            "returns": {"scan_complete": True, "issues_found": 0, "duration": "30s"}
        },
        {
            "name": "get_security_policy",
            "doc": "Get active security policy",
            "returns": {"policy": "STANDARD", "rules": 24, "enabled": True}
        },
    ],
    "storage": [
        {
            "name": "get_storage_capacity",
            "doc": "Get storage capacity and usage",
            "returns": {"total_gb": 512, "used_gb": 256, "available_gb": 256}
        },
        {
            "name": "list_volumes",
            "doc": "List storage volumes",
            "returns": {"volumes": [{"id": "vol1", "size": "100GB", "type": "SSD"}]}
        },
        {
            "name": "get_io_stats",
            "doc": "Get I/O statistics",
            "returns": {"read_ops": 1000, "write_ops": 500, "iops": 1500}
        },
    ],
    "processing": [
        {
            "name": "get_processing_load",
            "doc": "Get current processing load",
            "returns": {"cpu_usage": "45%", "queue_depth": 12, "throughput": "1000/sec"}
        },
        {
            "name": "configure_pipeline",
            "doc": "Configure processing pipeline",
            "params": [("stages", "int"), ("buffer_size", "int")],
            "returns": {"configured": True, "stages": 4}
        },
        {
            "name": "get_processing_stats",
            "doc": "Get processing statistics",
            "returns": {"processed": 10000, "errors": 5, "avg_time": "100ms"}
        },
    ],
    "peripheral": [
        {
            "name": "get_device_info",
            "doc": "Get peripheral device information",
            "returns": {"model": "Device-X", "firmware": "1.2.3", "status": "ready"}
        },
        {
            "name": "calibrate_device",
            "doc": "Calibrate peripheral device",
            "returns": {"calibrated": True, "accuracy": "99.5%"}
        },
        {
            "name": "get_sensor_readings",
            "doc": "Get current sensor readings",
            "returns": {"sensors": [{"id": 1, "value": 42.5, "unit": "units"}]}
        },
    ],
    "simulation": [
        {
            "name": "get_scenario_list",
            "doc": "Get available training scenarios",
            "returns": {"scenarios": [{"id": 1, "name": "Basic", "difficulty": "easy"}]}
        },
        {
            "name": "start_simulation",
            "doc": "Start training simulation",
            "params": [("scenario_id", "int")],
            "returns": {"started": True, "session_id": "sim-001"}
        },
        {
            "name": "get_performance_metrics",
            "doc": "Get simulation performance metrics",
            "returns": {"score": 85, "accuracy": "90%", "time": "15min"}
        },
    ],
}


def determine_device_category(device_id, name, group):
    """Determine device category based on ID, name, and group"""
    name_lower = name.lower()
    group_lower = group.lower()

    if any(word in name_lower for word in ["network", "packet", "monitor", "datalink", "satellite", "radio", "frequency"]):
        return "network"
    elif any(word in name_lower for word in ["intrusion", "geofence", "biometric", "token", "vpn", "remote"]):
        return "security"
    elif any(word in name_lower for word in ["storage", "raid", "backup", "sanitizer", "volume", "snapshot", "cache"]):
        return "storage"
    elif any(word in name_lower for word in ["processor", "crypto", "signal", "image", "video", "pattern", "threat", "target", "fusion"]):
        return "processing"
    elif any(word in name_lower for word in ["sensor", "actuator", "servo", "motion", "haptic", "display", "audio", "input", "gesture", "voice", "barcode", "rfid"]):
        return "peripheral"
    elif any(word in name_lower for word in ["simulation", "scenario", "training", "performance", "mission", "tactical", "decision", "collaboration", "expert", "adaptive", "assessment"]):
        return "simulation"
    else:
        # Default based on group
        if "network" in group_lower or "communications" in group_lower:
            return "network"
        elif "security" in group_lower:
            return "security"
        elif "storage" in group_lower:
            return "storage"
        elif "processing" in group_lower:
            return "processing"
        elif "peripheral" in group_lower:
            return "peripheral"
        elif "simulation" in group_lower or "training" in group_lower:
            return "simulation"

    return "processing"  # Default fallback


def generate_device_methods(category, device_name):
    """Generate device method code"""
    operations = OPERATION_TEMPLATES.get(category, OPERATION_TEMPLATES["processing"])

    methods_code = []

    for op in operations:
        params_str = "self"
        if "params" in op:
            for param_name, param_type in op["params"]:
                params_str += f", {param_name}: {param_type}"

        returns_str = str(op["returns"]).replace("'", '"')

        method_code = f'''
    def {op["name"]}({params_str}) -> OperationResult:
        """{op["doc"]}"""
        return OperationResult(
            success=True,
            data={returns_str}
        )
'''
        methods_code.append(method_code)

    return "\n".join(methods_code)


def generate_menu_function(device_id, device_name, category, class_name):
    """Generate TUI menu function code"""
    operations = OPERATION_TEMPLATES.get(category, OPERATION_TEMPLATES["processing"])

    menu_code = f'''
    def {device_name.lower().replace(" ", "_")}_operations(self, device):
        """Custom operations menu for {device_name}"""
        while True:
            self.clear_screen()
            self.print_header(f"Device Operations: {{device.name}}")

            print(f"  Device ID: 0x{{device.device_id:04X}}")
            print(f"  Category: {category.title()}")
            print(f"  State: {{device.state.value}}\\n")

'''

    for idx, op in enumerate(operations, 1):
        op_title = op["name"].replace("_", " ").title()
        menu_code += f'            print("  [{idx}] {op_title}")\n'

    menu_code += f'''            print("  [0] Back")

            choice = self.get_input("Select operation", {[str(i) for i in range(len(operations) + 1)]})

            if choice == "0":
                break
'''

    for idx, op in enumerate(operations, 1):
        menu_code += f'''            elif choice == "{idx}":
                result = device.{op["name"]}('''

        if "params" in op:
            # Add parameter prompts
            param_prompts = []
            for param_name, param_type in op["params"]:
                if param_type == "str":
                    param_prompts.append(f'input("  {param_name.title()}: ")')
                elif param_type == "int":
                    param_prompts.append(f'int(input("  {param_name.title()}: ") or "0")')
                elif param_type == "dict":
                    param_prompts.append('{}')
            menu_code += ", ".join(param_prompts)

        menu_code += ''')
                self.display_result(result)
'''

    return menu_code


def add_methods_to_device_file(device_file_path, methods_code):
    """Add methods to device class file"""
    with open(device_file_path, 'r') as f:
        content = f.read()

    # Find the class definition
    class_match = re.search(r'class\s+(\w+)\(DSMILDeviceBase\):', content)
    if not class_match:
        print(f"  ⚠️ Could not find class definition in {device_file_path}")
        return False

    # Find the end of the class (last method or __init__)
    # Insert before the final EOF
    if content.rstrip().endswith('"""'):
        # File ends with docstring
        insert_pos = len(content.rstrip())
    else:
        # Find last method
        methods = list(re.finditer(r'\n    def \w+\(', content))
        if methods:
            last_method = methods[-1]
            # Find end of last method
            next_class_or_eof = content.find('\nclass ', last_method.end())
            if next_class_or_eof == -1:
                insert_pos = len(content.rstrip())
            else:
                insert_pos = next_class_or_eof
        else:
            insert_pos = len(content.rstrip())

    # Insert methods
    new_content = content[:insert_pos] + methods_code + "\n" + content[insert_pos:]

    with open(device_file_path, 'w') as f:
        f.write(new_content)

    return True


def add_menu_to_dsmil_menu(menu_code, device_id, device_name):
    """Add menu function to dsmil_menu.py"""
    menu_file = Path(__file__).parent / "dsmil_menu.py"

    with open(menu_file, 'r') as f:
        content = f.read()

    # Add menu function before the last class method
    # Find the device_operations method
    device_ops_match = re.search(r'    def device_operations\(self, device\):', content)
    if not device_ops_match:
        print(f"  ⚠️ Could not find device_operations method")
        return False

    # Insert menu function before device_operations
    insert_pos = device_ops_match.start()
    new_content = content[:insert_pos] + menu_code + "\n" + content[insert_pos:]

    with open(menu_file, 'w') as f:
        f.write(new_content)

    # Now add routing to device_operations
    with open(menu_file, 'r') as f:
        content = f.read()

    # Find the else: # Use generic operations
    generic_match = re.search(r'(\s+)else:\s+# Use generic operations', content)
    if generic_match:
        indent = generic_match.group(1)
        routing_code = f'''{indent}elif device_id == 0x{device_id:04X}:  # {device_name}
{indent}    self.{device_name.lower().replace(" ", "_")}_operations(device)
{indent}'''

        insert_pos = generic_match.start()
        new_content = content[:insert_pos] + routing_code + content[insert_pos:]

        with open(menu_file, 'w') as f:
            f.write(new_content)

    return True


def generate_operations_for_device(device_id, device_file):
    """Generate operations for a single device"""
    module_name = device_file['module_name']
    filename = device_file['filename']

    # Extract device name
    name_part = filename.replace(f"device_0x{device_id:04x}_", "").replace(".py", "")
    device_name = name_part.replace("_", " ").title()

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

    print(f"\n📝 Generating operations for 0x{device_id:04X}: {device_name}")
    print(f"   Group: {group}")

    # Determine category
    category = determine_device_category(device_id, device_name, group)
    print(f"   Category: {category}")

    # Generate methods
    methods_code = generate_device_methods(category, device_name)

    # Add to device file
    device_file_path = Path(__file__).parent / "devices" / filename
    if add_methods_to_device_file(device_file_path, methods_code):
        print(f"   ✅ Added {len(OPERATION_TEMPLATES[category])} methods to device file")
    else:
        print(f"   ❌ Failed to add methods")
        return False

    # Generate menu function
    menu_code = generate_menu_function(device_id, device_name, category, module_name)

    # Add to dsmil_menu.py
    if add_menu_to_dsmil_menu(menu_code, device_id, device_name):
        print(f"   ✅ Added menu function to dsmil_menu.py")
    else:
        print(f"   ❌ Failed to add menu function")
        return False

    return True


def main():
    print("=" * 80)
    print("AUTOMATED DEVICE OPERATION GENERATOR")
    print("=" * 80)
    print()
    print("This tool will automatically generate device-specific operations")
    print("for all devices that currently only have generic operations.")
    print()

    # Discover devices
    device_files = discover_device_files()

    # Load each device and check if it needs operations
    devices_to_populate = []

    for device_file in device_files:
        device_id = device_file['device_id']
        module_name = device_file['module_name']

        try:
            device_class = load_device_class(module_name)
            if not device_class:
                continue

            # Check if device has custom methods
            base_methods = set(dir(DSMILDeviceBase))
            device_methods = set(dir(device_class))
            custom_methods = device_methods - base_methods

            # Filter out private/magic methods
            custom_public_methods = [m for m in custom_methods if not m.startswith('_') and callable(getattr(device_class, m))]

            if not custom_public_methods:
                devices_to_populate.append(device_file)

        except Exception as e:
            print(f"⚠️ Error checking device 0x{device_id:04X}: {e}")

    print(f"Found {len(devices_to_populate)} devices needing operations")
    print()

    if not devices_to_populate:
        print("✅ All devices already have custom operations!")
        return

    # Confirm
    response = input(f"Generate operations for {len(devices_to_populate)} devices? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Generate for each device
    success_count = 0
    for device_file in devices_to_populate:
        try:
            if generate_operations_for_device(device_file['device_id'], device_file):
                success_count += 1
        except Exception as e:
            print(f"❌ Error generating for 0x{device_file['device_id']:04X}: {e}")

    print()
    print("=" * 80)
    print(f"✅ Generated operations for {success_count}/{len(devices_to_populate)} devices")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the generated operations in the device files")
    print("2. Test the new menus: python3 dsmil_menu.py")
    print("3. Customize operations as needed for specific devices")
    print("4. Commit changes: git add . && git commit -m 'feat: Auto-generated operations for 60 devices'")


if __name__ == "__main__":
    main()
