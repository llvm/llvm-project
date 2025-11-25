#!/usr/bin/env python3
"""
⚠️  DEPRECATED - This component will be removed in v3.0.0 (2026 Q3)
⚠️  Use: DSMILIntegrationAdapter.discover_all_devices_cascading()
⚠️  See DEPRECATION_PLAN.md for migration guide

DSMIL Local System Discovery Script

Discovers DSMIL devices, drivers, and framework components on the local system.
Performs comprehensive hardware and software enumeration to identify what's
actually present vs what's integrated in the framework.

Author: DSMIL Integration Framework
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import sys
import os
import subprocess
import re
import glob
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from dsmil_auto_discover import list_devices, get_integration_summary


# ============================================================================
# DEPRECATION WARNING - REMOVE IN v3.0.0 (2026 Q3)
# ============================================================================
import warnings
warnings.warn(
    "\n" + "=" * 80 + "\n"
    "⚠️  DEPRECATED: dsmil_discover.py\n\n"
    "This component is deprecated and will be removed in v3.0.0 (2026 Q3).\n\n"
    "Migration:\n"
    "  from dsmil_integration_adapter import DSMILIntegrationAdapter\n"
    "  adapter = DSMILIntegrationAdapter()\n"
    "  devices = adapter.discover_all_devices_cascading()\n\n"
    "See DEPRECATION_PLAN.md for complete migration guide.\n"
    + "=" * 80,
    DeprecationWarning,
    stacklevel=2
)
# ============================================================================


class DSMILDiscovery:
    """Local system discovery for DSMIL devices and drivers"""

    def __init__(self):
        self.discoveries = {
            'kernel_modules': [],
            'device_nodes': [],
            'dmesg_references': [],
            'pci_devices': [],
            'acpi_tables': [],
            'smi_interface': {},
            'processes': [],
            'firmware_files': [],
            'sys_devices': [],
            'proc_devices': [],
            'mmio_regions': [],
            'wmi_interfaces': [],
            'efi_variables': [],
            'usb_devices': [],
            'dell_tools': [],
            'bios_settings': {},
            'intel_me': {},
        }
        self.integrated_devices = []
        self.framework_status = {}

    def run_command(self, cmd: str, shell: bool = False) -> Tuple[int, str, str]:
        """Run a system command and return result"""
        try:
            if shell:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def check_root_privileges(self) -> bool:
        """Check if running with root privileges"""
        return os.geteuid() == 0

    def discover_kernel_modules(self):
        """Discover loaded kernel modules related to DSMIL"""
        print("  [*] Checking kernel modules...")

        # Check lsmod output
        ret, stdout, _ = self.run_command("lsmod")
        if ret == 0:
            lines = stdout.strip().split('\n')[1:]  # Skip header

            # Look for DSMIL-related modules
            keywords = ['dsmil', 'dell', 'smm', 'dcdbas', 'wmi', 'smi']

            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    module_name = parts[0]
                    size = parts[1]
                    used_by = parts[2] if len(parts) > 2 else "0"

                    # Check if module name contains any keywords
                    if any(kw in module_name.lower() for kw in keywords):
                        self.discoveries['kernel_modules'].append({
                            'name': module_name,
                            'size': size,
                            'used_by': used_by,
                            'relevance': 'high'
                        })

        # Check for specific Dell modules
        dell_modules = ['dcdbas', 'dell_smm_hwmon', 'dell_wmi', 'dell_smbios']
        for module in dell_modules:
            ret, stdout, _ = self.run_command(f"modinfo {module}")
            if ret == 0:
                # Module exists even if not loaded
                if not any(m['name'] == module for m in self.discoveries['kernel_modules']):
                    self.discoveries['kernel_modules'].append({
                        'name': module,
                        'size': 'N/A',
                        'used_by': 'not loaded',
                        'relevance': 'available'
                    })

        print(f"      Found {len(self.discoveries['kernel_modules'])} relevant modules")

    def discover_device_nodes(self):
        """Discover device nodes in /dev"""
        print("  [*] Checking device nodes...")

        # Check /dev for DSMIL-related devices
        dev_patterns = [
            '/dev/smi*',
            '/dev/dsmil*',
            '/dev/dell*',
            '/dev/dcdbas*',
            '/dev/smm*',
        ]

        for pattern in dev_patterns:
            devices = glob.glob(pattern)
            for device in devices:
                stat_info = os.stat(device)
                self.discoveries['device_nodes'].append({
                    'path': device,
                    'major': os.major(stat_info.st_rdev),
                    'minor': os.minor(stat_info.st_rdev),
                    'permissions': oct(stat_info.st_mode)[-4:],
                })

        # Check for character devices with specific major numbers
        # Dell DCDBAS typically uses major 10 (misc)
        if os.path.exists('/dev'):
            try:
                ret, stdout, _ = self.run_command("ls -l /dev", shell=True)
                if ret == 0:
                    for line in stdout.split('\n'):
                        if 'dell' in line.lower() or 'smi' in line.lower():
                            parts = line.split()
                            if len(parts) >= 10:
                                self.discoveries['device_nodes'].append({
                                    'path': parts[-1],
                                    'info': ' '.join(parts[:-1]),
                                    'source': 'ls scan'
                                })
            except:
                pass

        print(f"      Found {len(self.discoveries['device_nodes'])} device nodes")

    def discover_dmesg_references(self):
        """Search dmesg for DSMIL/Dell references"""
        print("  [*] Checking kernel messages (dmesg)...")

        ret, stdout, stderr = self.run_command("dmesg", shell=False)

        if ret != 0:
            # Try with sudo
            ret, stdout, stderr = self.run_command("sudo dmesg", shell=True)

        if ret == 0:
            lines = stdout.strip().split('\n')

            keywords = ['dsmil', 'dell', 'smi', 'dcdbas', 'smm', 'tpm', 'latitude']

            for line in lines:
                if any(kw in line.lower() for kw in keywords):
                    # Extract timestamp if present
                    timestamp_match = re.match(r'\[\s*(\d+\.\d+)\]', line)
                    timestamp = timestamp_match.group(1) if timestamp_match else 'unknown'

                    self.discoveries['dmesg_references'].append({
                        'timestamp': timestamp,
                        'message': line,
                    })

        # Limit to most recent 50 entries
        self.discoveries['dmesg_references'] = self.discoveries['dmesg_references'][-50:]

        print(f"      Found {len(self.discoveries['dmesg_references'])} relevant messages")

    def discover_pci_devices(self):
        """Discover PCI devices"""
        print("  [*] Checking PCI devices...")

        ret, stdout, _ = self.run_command("lspci")
        if ret == 0:
            lines = stdout.strip().split('\n')

            keywords = ['dell', 'system', 'management', 'tpm', 'security']

            for line in lines:
                if any(kw in line.lower() for kw in keywords):
                    parts = line.split(' ', 1)
                    pci_id = parts[0] if len(parts) > 0 else 'unknown'
                    description = parts[1] if len(parts) > 1 else line

                    self.discoveries['pci_devices'].append({
                        'id': pci_id,
                        'description': description,
                    })

        print(f"      Found {len(self.discoveries['pci_devices'])} relevant PCI devices")

    def discover_acpi_tables(self):
        """Discover ACPI tables"""
        print("  [*] Checking ACPI tables...")

        acpi_paths = [
            '/sys/firmware/acpi/tables/',
            '/sys/firmware/efi/systab',
        ]

        for path in acpi_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    try:
                        tables = os.listdir(path)
                        for table in tables:
                            table_path = os.path.join(path, table)
                            if os.path.isfile(table_path):
                                size = os.path.getsize(table_path)
                                self.discoveries['acpi_tables'].append({
                                    'name': table,
                                    'path': table_path,
                                    'size': size,
                                })
                    except PermissionError:
                        pass

        print(f"      Found {len(self.discoveries['acpi_tables'])} ACPI tables")

    def discover_smi_interface(self):
        """Check SMI interface availability"""
        print("  [*] Checking SMI interface (ports 0xB2/0xB3)...")

        # Check if /dev/port exists (direct port access)
        has_dev_port = os.path.exists('/dev/port')

        # Check if ioperm/iopl capabilities are available
        has_root = self.check_root_privileges()

        # Check for SMI-related kernel parameters
        ret, stdout, _ = self.run_command("cat /proc/cmdline", shell=True)
        cmdline = stdout.strip() if ret == 0 else ""

        self.discoveries['smi_interface'] = {
            'dev_port_exists': has_dev_port,
            'root_privileges': has_root,
            'kernel_cmdline': cmdline,
            'port_b2_accessible': has_dev_port and has_root,
            'port_b3_accessible': has_dev_port and has_root,
        }

        # Check if SMI is disabled
        smi_disabled = 'nosmi' in cmdline.lower()
        self.discoveries['smi_interface']['smi_disabled'] = smi_disabled

        status = "Available" if (has_dev_port and has_root) else "Limited"
        print(f"      SMI Interface: {status}")

    def discover_processes(self):
        """Discover DSMIL-related processes"""
        print("  [*] Checking running processes...")

        ret, stdout, _ = self.run_command("ps aux", shell=True)
        if ret == 0:
            lines = stdout.strip().split('\n')

            keywords = ['dsmil', 'dell', 'smi', 'dcdbas']

            for line in lines[1:]:  # Skip header
                if any(kw in line.lower() for kw in keywords):
                    parts = line.split(None, 10)
                    if len(parts) >= 11:
                        self.discoveries['processes'].append({
                            'user': parts[0],
                            'pid': parts[1],
                            'cpu': parts[2],
                            'mem': parts[3],
                            'command': parts[10],
                        })

        print(f"      Found {len(self.discoveries['processes'])} relevant processes")

    def discover_firmware_files(self):
        """Discover firmware files"""
        print("  [*] Checking firmware files...")

        firmware_paths = [
            '/lib/firmware/dell*',
            '/lib/firmware/dsmil*',
        ]

        for pattern in firmware_paths:
            files = glob.glob(pattern)
            for filepath in files:
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    self.discoveries['firmware_files'].append({
                        'path': filepath,
                        'size': size,
                        'name': os.path.basename(filepath),
                    })

        print(f"      Found {len(self.discoveries['firmware_files'])} firmware files")

    def discover_sys_devices(self):
        """Discover devices in /sys"""
        print("  [*] Checking /sys devices...")

        sys_patterns = [
            '/sys/devices/platform/dcdbas*',
            '/sys/devices/platform/dell*',
            '/sys/class/dmi/id/*',
        ]

        for pattern in sys_patterns:
            paths = glob.glob(pattern)
            for path in paths:
                if os.path.exists(path):
                    self.discoveries['sys_devices'].append({
                        'path': path,
                        'type': 'directory' if os.path.isdir(path) else 'file',
                    })

        # Read DMI information
        dmi_path = '/sys/class/dmi/id/'
        if os.path.exists(dmi_path):
            dmi_files = ['sys_vendor', 'product_name', 'product_version', 'bios_version']
            dmi_info = {}
            for filename in dmi_files:
                filepath = os.path.join(dmi_path, filename)
                if os.path.isfile(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            dmi_info[filename] = f.read().strip()
                    except:
                        pass

            if dmi_info:
                self.discoveries['sys_devices'].append({
                    'path': dmi_path,
                    'type': 'dmi_info',
                    'data': dmi_info,
                })

        print(f"      Found {len(self.discoveries['sys_devices'])} /sys entries")

    def discover_proc_devices(self):
        """Discover devices in /proc"""
        print("  [*] Checking /proc devices...")

        # Check /proc/devices for character/block devices
        if os.path.exists('/proc/devices'):
            try:
                with open('/proc/devices', 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if 'dell' in line.lower() or 'smi' in line.lower():
                            self.discoveries['proc_devices'].append({
                                'entry': line.strip(),
                                'source': '/proc/devices',
                            })
            except:
                pass

        # Check /proc/modules (alternative to lsmod)
        if os.path.exists('/proc/modules'):
            try:
                with open('/proc/modules', 'r') as f:
                    for line in f:
                        if any(kw in line.lower() for kw in ['dell', 'smi', 'dcdbas']):
                            parts = line.split()
                            if parts:
                                self.discoveries['proc_devices'].append({
                                    'entry': parts[0],
                                    'source': '/proc/modules',
                                })
            except:
                pass

        print(f"      Found {len(self.discoveries['proc_devices'])} /proc entries")

    def discover_mmio_regions(self):
        """Discover memory-mapped I/O regions"""
        print("  [*] Checking MMIO regions in /proc/iomem...")

        if os.path.exists('/proc/iomem'):
            try:
                with open('/proc/iomem', 'r') as f:
                    for line in f:
                        # Look for Dell/SMM/DSMIL related regions
                        if any(kw in line.lower() for kw in ['dell', 'smm', 'smi', 'dsmil', 'smram']):
                            self.discoveries['mmio_regions'].append({
                                'entry': line.strip(),
                                'type': 'memory',
                            })
                        # Also capture reserved regions that might be SMM
                        elif 'reserved' in line.lower() and ('a0000' in line or 'e0000' in line or 'tseg' in line.lower()):
                            self.discoveries['mmio_regions'].append({
                                'entry': line.strip(),
                                'type': 'reserved_smm_candidate',
                            })
            except:
                pass

        # Check for SMRAM in specific locations
        if os.path.exists('/proc/ioports'):
            try:
                with open('/proc/ioports', 'r') as f:
                    for line in f:
                        if any(kw in line.lower() for kw in ['b2-b3', 'smi', 'smm']):
                            self.discoveries['mmio_regions'].append({
                                'entry': line.strip(),
                                'type': 'ioport',
                            })
            except:
                pass

        print(f"      Found {len(self.discoveries['mmio_regions'])} MMIO/SMRAM regions")

    def discover_msr_registers(self):
        """Discover and read MSR (Model Specific Registers)"""
        print("  [*] Checking MSR registers...")

        # Check if msr module is loaded
        ret, stdout, _ = self.run_command("lsmod")
        msr_loaded = 'msr' in stdout if ret == 0 else False

        if not msr_loaded:
            # Try to load msr module
            if self.check_root_privileges():
                ret, _, _ = self.run_command("modprobe msr", shell=True)
                msr_loaded = (ret == 0)

        msr_devices = glob.glob('/dev/cpu/*/msr')

        self.discoveries['mmio_regions'].append({
            'entry': f'MSR module loaded: {msr_loaded}, MSR devices: {len(msr_devices)}',
            'type': 'msr_status',
        })

        # Try to read key SMM-related MSRs (requires root and msr module)
        if msr_loaded and msr_devices and self.check_root_privileges():
            # Important MSRs for SMM detection:
            # 0x9E - IA32_SMBASE (SMM base address)
            # 0x1A0 - IA32_MISC_ENABLE
            # 0x3A - IA32_FEATURE_CONTROL
            # 0x17 - IA32_PLATFORM_ID
            important_msrs = {
                '0x9e': 'IA32_SMBASE (SMM Base)',
                '0x1a0': 'IA32_MISC_ENABLE',
                '0x3a': 'IA32_FEATURE_CONTROL',
                '0x17': 'IA32_PLATFORM_ID',
                '0xce': 'MSR_PLATFORM_INFO',
            }

            # Use rdmsr if available
            ret, stdout, _ = self.run_command("which rdmsr")
            if ret == 0:
                for msr_addr, msr_name in important_msrs.items():
                    ret, stdout, _ = self.run_command(f"rdmsr {msr_addr}", shell=True)
                    if ret == 0:
                        self.discoveries['mmio_regions'].append({
                            'entry': f'{msr_name} = 0x{stdout.strip()}',
                            'type': 'msr_value',
                        })

        print(f"      MSR module: {'Loaded' if msr_loaded else 'Not loaded'}, Devices: {len(msr_devices)}")

    def discover_wmi_interfaces(self):
        """Discover Dell WMI (Windows Management Instrumentation) interfaces"""
        print("  [*] Checking Dell WMI interfaces...")

        # Check for Dell WMI kernel modules
        wmi_modules = ['dell_wmi', 'dell_wmi_descriptor', 'dell_smbios_wmi']
        for module in wmi_modules:
            ret, stdout, _ = self.run_command(f"modinfo {module}")
            if ret == 0:
                self.discoveries['wmi_interfaces'].append({
                    'module': module,
                    'status': 'available',
                })

        # Check /sys/devices/platform for WMI devices
        wmi_paths = glob.glob('/sys/devices/platform/PNP0C14:*') + \
                   glob.glob('/sys/devices/platform/dell-wmi*')

        for path in wmi_paths:
            if os.path.exists(path):
                self.discoveries['wmi_interfaces'].append({
                    'path': path,
                    'type': 'platform_device',
                })

        # Check for WMI character devices
        wmi_devs = glob.glob('/dev/wmi*')
        for dev in wmi_devs:
            self.discoveries['wmi_interfaces'].append({
                'device': dev,
                'type': 'character_device',
            })

        print(f"      Found {len(self.discoveries['wmi_interfaces'])} WMI interfaces")

    def discover_efi_variables(self):
        """Discover EFI/UEFI variables related to Dell/DSMIL"""
        print("  [*] Checking EFI variables...")

        efi_paths = [
            '/sys/firmware/efi/efivars',
            '/sys/firmware/efi/vars',
        ]

        for efi_path in efi_paths:
            if os.path.exists(efi_path):
                try:
                    # List EFI variables
                    vars = os.listdir(efi_path)
                    dell_vars = [v for v in vars if 'dell' in v.lower() or 'dsmil' in v.lower()]

                    for var in dell_vars:
                        var_path = os.path.join(efi_path, var)
                        try:
                            size = os.path.getsize(var_path)
                            self.discoveries['efi_variables'].append({
                                'name': var,
                                'path': var_path,
                                'size': size,
                            })
                        except:
                            pass
                except PermissionError:
                    self.discoveries['efi_variables'].append({
                        'name': 'Permission denied',
                        'path': efi_path,
                        'note': 'Requires root access',
                    })

        # Check EFI system table
        if os.path.exists('/sys/firmware/efi/systab'):
            try:
                with open('/sys/firmware/efi/systab', 'r') as f:
                    content = f.read()
                    if any(kw in content.lower() for kw in ['dell', 'smbios']):
                        self.discoveries['efi_variables'].append({
                            'name': 'systab',
                            'type': 'system_table',
                            'content': content[:200],
                        })
            except:
                pass

        print(f"      Found {len(self.discoveries['efi_variables'])} EFI variables")

    def discover_usb_devices(self):
        """Discover USB devices from Dell"""
        print("  [*] Checking USB devices...")

        ret, stdout, _ = self.run_command("lsusb")
        if ret == 0:
            lines = stdout.strip().split('\n')
            for line in lines:
                # Dell USB vendor ID is 0x413c
                if 'dell' in line.lower() or '413c' in line.lower():
                    parts = line.split(':', 1)
                    bus_dev = parts[0] if len(parts) > 0 else ''
                    description = parts[1].strip() if len(parts) > 1 else line

                    self.discoveries['usb_devices'].append({
                        'bus_device': bus_dev,
                        'description': description,
                    })

        # Check USB device details in /sys
        usb_paths = glob.glob('/sys/bus/usb/devices/*/manufacturer')
        for path in usb_paths:
            try:
                with open(path, 'r') as f:
                    manufacturer = f.read().strip()
                    if 'dell' in manufacturer.lower():
                        device_path = os.path.dirname(path)
                        # Read product name
                        product_path = os.path.join(device_path, 'product')
                        product = ''
                        if os.path.exists(product_path):
                            with open(product_path, 'r') as pf:
                                product = pf.read().strip()

                        self.discoveries['usb_devices'].append({
                            'manufacturer': manufacturer,
                            'product': product,
                            'path': device_path,
                        })
            except:
                pass

        print(f"      Found {len(self.discoveries['usb_devices'])} Dell USB devices")

    def discover_dell_tools(self):
        """Discover installed Dell management tools"""
        print("  [*] Checking for Dell management tools...")

        # Check for common Dell utilities
        dell_tools = [
            'dcdbas',
            'srvadmin-services',
            'omreport',
            'racadm',
            'syscfg',
            'dset',
            'dell-system-update',
        ]

        for tool in dell_tools:
            ret, stdout, _ = self.run_command(f"which {tool}")
            if ret == 0:
                path = stdout.strip()
                self.discoveries['dell_tools'].append({
                    'name': tool,
                    'path': path,
                    'status': 'installed',
                })

        # Check for Dell packages
        ret, stdout, _ = self.run_command("dpkg -l | grep -i dell", shell=True)
        if ret == 0:
            for line in stdout.strip().split('\n'):
                if line:
                    self.discoveries['dell_tools'].append({
                        'name': line.split()[1] if len(line.split()) > 1 else 'unknown',
                        'type': 'package',
                        'status': 'installed',
                    })

        # Check for Dell services
        ret, stdout, _ = self.run_command("systemctl list-units --all | grep -i dell", shell=True)
        if ret == 0:
            for line in stdout.strip().split('\n'):
                if line and 'dell' in line.lower():
                    parts = line.split()
                    if parts:
                        self.discoveries['dell_tools'].append({
                            'name': parts[0],
                            'type': 'service',
                            'status': parts[2] if len(parts) > 2 else 'unknown',
                        })

        print(f"      Found {len(self.discoveries['dell_tools'])} Dell tools/packages")

    def discover_bios_settings(self):
        """Discover BIOS/UEFI settings related to SMM/SMI"""
        print("  [*] Checking BIOS settings...")

        # Check DMI BIOS information
        dmi_bios_path = '/sys/class/dmi/id'
        if os.path.exists(dmi_bios_path):
            bios_files = {
                'bios_vendor': 'BIOS Vendor',
                'bios_version': 'BIOS Version',
                'bios_date': 'BIOS Date',
                'bios_release': 'BIOS Release',
            }

            for filename, description in bios_files.items():
                filepath = os.path.join(dmi_bios_path, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            value = f.read().strip()
                            self.discoveries['bios_settings'][description] = value
                    except:
                        pass

        # Check for SMM/SMI related kernel boot parameters
        if os.path.exists('/proc/cmdline'):
            try:
                with open('/proc/cmdline', 'r') as f:
                    cmdline = f.read().strip()
                    smi_params = []
                    if 'nosmi' in cmdline:
                        smi_params.append('nosmi (SMI disabled)')
                    if 'nmi_watchdog' in cmdline:
                        smi_params.append('nmi_watchdog enabled')
                    if 'intel_iommu' in cmdline:
                        smi_params.append('intel_iommu enabled')

                    self.discoveries['bios_settings']['Kernel Parameters'] = ', '.join(smi_params) if smi_params else 'No SMI-related params'
            except:
                pass

        # Try dmidecode for detailed BIOS info (requires root)
        if self.check_root_privileges():
            ret, stdout, _ = self.run_command("dmidecode -t bios", shell=True)
            if ret == 0:
                self.discoveries['bios_settings']['DMI Decode'] = 'Available (see dmesg)'

        print(f"      BIOS Info: {len(self.discoveries['bios_settings'])} settings found")

    def discover_intel_me(self):
        """Discover Intel Management Engine (ME) and HAP bit status"""
        print("  [*] Checking Intel ME and HAP bit...")

        me_info = {
            'detected': False,
            'modules': [],
            'devices': [],
            'version': 'Unknown',
            'hap_bit': 'Unknown',
            'operational_state': 'Unknown',
            'amt_enabled': 'Unknown',
        }

        # Check for MEI (ME Interface) kernel modules
        mei_modules = ['mei', 'mei_me', 'mei_wdt', 'mei_hdcp', 'mei_amt']
        for module in mei_modules:
            ret, stdout, _ = self.run_command(f"lsmod | grep {module}", shell=True)
            if ret == 0 and stdout.strip():
                me_info['modules'].append(module)
                me_info['detected'] = True

        # Check for MEI device nodes
        mei_devices = glob.glob('/dev/mei*')
        me_info['devices'] = [os.path.basename(d) for d in mei_devices]
        if mei_devices:
            me_info['detected'] = True

        # Check /sys for MEI information
        mei_paths = glob.glob('/sys/bus/pci/drivers/mei_me/*/mei/mei*')
        if mei_paths:
            me_info['detected'] = True
            # Try to read MEI device attributes
            for path in mei_paths[:1]:  # Check first device
                # Read ME version if available
                version_path = os.path.join(path, 'fw_ver')
                if os.path.exists(version_path):
                    try:
                        with open(version_path, 'r') as f:
                            me_info['version'] = f.read().strip()
                    except:
                        pass

                # Read operational state
                state_path = os.path.join(path, 'fw_status')
                if os.path.exists(state_path):
                    try:
                        with open(state_path, 'r') as f:
                            state_val = f.read().strip()
                            me_info['operational_state'] = state_val
                    except:
                        pass

        # Check PCI devices for Intel MEI controller
        ret, stdout, _ = self.run_command("lspci -d 8086:")
        if ret == 0:
            for line in stdout.split('\n'):
                if any(kw in line.lower() for kw in ['mei', 'management engine', 'amt', 'csme']):
                    me_info['detected'] = True
                    pci_id = line.split()[0] if line.split() else 'Unknown'
                    me_info['pci_device'] = line.strip()

        # Try to detect HAP bit status using various methods
        hap_detected = False

        # Method 1: Check via intelmetool if available
        ret, stdout, _ = self.run_command("which intelmetool")
        if ret == 0:
            ret, stdout, _ = self.run_command("sudo intelmetool -m", shell=True)
            if ret == 0:
                if 'hap' in stdout.lower():
                    if 'set' in stdout.lower() or 'enabled' in stdout.lower():
                        me_info['hap_bit'] = 'SET (ME Disabled)'
                        hap_detected = True
                    elif 'not set' in stdout.lower() or 'disabled' in stdout.lower():
                        me_info['hap_bit'] = 'NOT SET (ME Enabled)'
                        hap_detected = True

        # Method 2: Check via me_cleaner detection
        ret, stdout, _ = self.run_command("which me_cleaner.py")
        if ret == 0 and not hap_detected:
            # me_cleaner can read flash and detect HAP bit
            # But requires flash access - skip actual read
            me_info['hap_bit'] += ' (me_cleaner available for checking)'

        # Method 3: Check MEI device status as indicator
        if me_info['operational_state'] != 'Unknown' and not hap_detected:
            # If operational_state shows certain patterns, ME is active
            if 'normal' in me_info['operational_state'].lower():
                me_info['hap_bit'] = 'Likely NOT SET (ME operational)'
            elif 'disable' in me_info['operational_state'].lower():
                me_info['hap_bit'] = 'Possibly SET (ME disabled)'

        # Method 4: Check dmesg for MEI/ME messages
        ret, stdout, _ = self.run_command("dmesg | grep -i 'mei\\|management engine'", shell=True)
        if ret == 0 and stdout.strip():
            me_messages = stdout.strip().split('\n')[-5:]  # Last 5 messages
            me_info['dmesg_messages'] = me_messages

            # Look for HAP-related messages
            for msg in me_messages:
                if 'hap' in msg.lower():
                    if 'disable' in msg.lower() or 'off' in msg.lower():
                        me_info['hap_bit'] = 'SET via dmesg (ME Disabled)'
                        break

        # Check for AMT (Active Management Technology)
        # AMT uses ports 16992, 16993 (HTTP/HTTPS)
        ret, stdout, _ = self.run_command("netstat -tuln | grep -E '16992|16993'", shell=True)
        if ret == 0 and stdout.strip():
            me_info['amt_enabled'] = 'YES (Ports 16992/16993 listening)'
        else:
            me_info['amt_enabled'] = 'NO (AMT ports not listening)'

        # Check for AMT via WMI
        amt_wmi_path = '/sys/devices/platform/amt_wmi*'
        if glob.glob(amt_wmi_path):
            me_info['amt_enabled'] = 'YES (WMI interface detected)'

        # Store results
        self.discoveries['intel_me'] = me_info

        if me_info['detected']:
            print(f"      Intel ME: DETECTED")
            print(f"      HAP Bit: {me_info['hap_bit']}")
            print(f"      Modules: {len(me_info['modules'])}, Devices: {len(me_info['devices'])}")
        else:
            print(f"      Intel ME: Not detected or disabled")

        # Add security recommendation
        if me_info['detected'] and me_info['hap_bit'] == 'Unknown':
            print(f"      [!] RECOMMENDATION: Check HAP bit status with 'sudo intelmetool -m'")
        elif 'NOT SET' in me_info['hap_bit']:
            print(f"      [!] SECURITY: HAP bit not set - Intel ME is ACTIVE")

    def get_framework_status(self):
        """Get DSMIL framework integration status"""
        print("  [*] Checking framework integration status...")

        try:
            summary = get_integration_summary()
            self.framework_status = summary
            self.integrated_devices = list_devices()
            print(f"      Framework: {summary['total_registered']} devices registered")
        except Exception as e:
            print(f"      Framework error: {e}")
            self.framework_status = {'error': str(e)}

    def run_discovery(self):
        """Run all discovery checks"""
        print("\n" + "=" * 80)
        print("DSMIL Local System Discovery - DEEP SCAN")
        print("=" * 80)
        print(f"\nStarting discovery at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Check privileges
        if not self.check_root_privileges():
            print("  [!] WARNING: Not running as root. Some checks may be limited.")
            print("  [!] For full MSR/MMIO access, run with: sudo python3 dsmil_discover.py\n")

        # Run all discovery methods
        print("━━━ Basic Discovery ━━━")
        self.discover_kernel_modules()
        self.discover_device_nodes()
        self.discover_dmesg_references()
        self.discover_pci_devices()
        self.discover_acpi_tables()
        self.discover_smi_interface()
        self.discover_processes()
        self.discover_firmware_files()
        self.discover_sys_devices()
        self.discover_proc_devices()

        print("\n━━━ Deep Hardware Discovery ━━━")
        self.discover_mmio_regions()
        self.discover_msr_registers()
        self.discover_intel_me()
        self.discover_wmi_interfaces()
        self.discover_efi_variables()
        self.discover_usb_devices()
        self.discover_dell_tools()
        self.discover_bios_settings()

        print("\n━━━ Framework Status ━━━")
        self.get_framework_status()

        print("\n" + "=" * 80)
        print("Discovery complete!")
        print("=" * 80)

    def print_report(self):
        """Print comprehensive discovery report"""
        print("\n" + "=" * 80)
        print("DISCOVERY REPORT")
        print("=" * 80)

        # System Information
        if self.discoveries['sys_devices']:
            dmi_entry = next((d for d in self.discoveries['sys_devices']
                            if d.get('type') == 'dmi_info'), None)
            if dmi_entry and 'data' in dmi_entry:
                print("\n━━━ System Information ━━━")
                dmi = dmi_entry['data']
                print(f"  Vendor:          {dmi.get('sys_vendor', 'Unknown')}")
                print(f"  Product:         {dmi.get('product_name', 'Unknown')}")
                print(f"  Version:         {dmi.get('product_version', 'Unknown')}")
                print(f"  BIOS Version:    {dmi.get('bios_version', 'Unknown')}")

        # Framework Status
        if self.framework_status:
            print("\n━━━ DSMIL Framework Status ━━━")
            if 'error' in self.framework_status:
                print(f"  Status: Error - {self.framework_status['error']}")
            else:
                print(f"  Integration:     {self.framework_status.get('integration_name', 'Unknown')}")
                print(f"  Version:         {self.framework_status.get('version', 'Unknown')}")
                print(f"  Registered:      {self.framework_status.get('total_registered', 0)} devices")
                print(f"  Enabled:         {self.framework_status.get('enabled', 0)} devices")
                print(f"  Initialized:     {self.framework_status.get('initialized', 0)} devices")

        # Kernel Modules
        if self.discoveries['kernel_modules']:
            print(f"\n━━━ Kernel Modules ({len(self.discoveries['kernel_modules'])}) ━━━")
            for mod in self.discoveries['kernel_modules'][:10]:
                status = "LOADED" if mod['used_by'] != 'not loaded' else "AVAILABLE"
                print(f"  [{status:9}] {mod['name']:20} Size: {mod['size']:10} Used by: {mod['used_by']}")
            if len(self.discoveries['kernel_modules']) > 10:
                print(f"  ... and {len(self.discoveries['kernel_modules']) - 10} more")

        # Device Nodes
        if self.discoveries['device_nodes']:
            print(f"\n━━━ Device Nodes ({len(self.discoveries['device_nodes'])}) ━━━")
            for dev in self.discoveries['device_nodes']:
                if 'major' in dev:
                    print(f"  {dev['path']:30} Major: {dev['major']:3} Minor: {dev['minor']:3} Perms: {dev['permissions']}")
                else:
                    print(f"  {dev.get('path', 'Unknown')}")

        # SMI Interface
        if self.discoveries['smi_interface']:
            print("\n━━━ SMI Interface Status ━━━")
            smi = self.discoveries['smi_interface']
            print(f"  /dev/port exists:    {smi.get('dev_port_exists', False)}")
            print(f"  Root privileges:     {smi.get('root_privileges', False)}")
            print(f"  SMI disabled:        {smi.get('smi_disabled', 'Unknown')}")
            print(f"  Port 0xB2 access:    {smi.get('port_b2_accessible', False)}")
            print(f"  Port 0xB3 access:    {smi.get('port_b3_accessible', False)}")

        # PCI Devices
        if self.discoveries['pci_devices']:
            print(f"\n━━━ Relevant PCI Devices ({len(self.discoveries['pci_devices'])}) ━━━")
            for pci in self.discoveries['pci_devices']:
                print(f"  {pci['id']:10} {pci['description']}")

        # Running Processes
        if self.discoveries['processes']:
            print(f"\n━━━ Relevant Processes ({len(self.discoveries['processes'])}) ━━━")
            for proc in self.discoveries['processes']:
                print(f"  PID {proc['pid']:6} {proc['user']:10} {proc['command'][:60]}")

        # Firmware Files
        if self.discoveries['firmware_files']:
            print(f"\n━━━ Firmware Files ({len(self.discoveries['firmware_files'])}) ━━━")
            for fw in self.discoveries['firmware_files']:
                print(f"  {fw['name']:30} {fw['size']:10} bytes")

        # Recent dmesg entries
        if self.discoveries['dmesg_references']:
            print(f"\n━━━ Recent Kernel Messages (showing last 10 of {len(self.discoveries['dmesg_references'])}) ━━━")
            for msg in self.discoveries['dmesg_references'][-10:]:
                print(f"  [{msg['timestamp']:>12}] {msg['message'][:100]}")

        # Intel ME & HAP Bit
        if self.discoveries['intel_me']:
            print("\n━━━ Intel Management Engine (ME) & HAP Bit ━━━")
            me = self.discoveries['intel_me']
            print(f"  Detected:            {me.get('detected', False)}")
            print(f"  ME Version:          {me.get('version', 'Unknown')}")
            print(f"  HAP Bit:             {me.get('hap_bit', 'Unknown')}")
            print(f"  Operational State:   {me.get('operational_state', 'Unknown')}")
            print(f"  AMT Enabled:         {me.get('amt_enabled', 'Unknown')}")
            if me.get('modules'):
                print(f"  Loaded Modules:      {', '.join(me['modules'][:5])}")
            if me.get('devices'):
                print(f"  Device Nodes:        {', '.join(me['devices'][:3])}")
            if me.get('pci_device'):
                print(f"  PCI Device:          {me['pci_device']}")

            # Security warning
            if me.get('detected') and 'NOT SET' in me.get('hap_bit', ''):
                print(f"\n  ⚠️  SECURITY WARNING: HAP bit not set - Intel ME is ACTIVE")
                print(f"  ⚠️  For MIL-SPEC systems, consider setting HAP bit to disable ME")
            elif me.get('detected') and 'SET' in me.get('hap_bit', ''):
                print(f"\n  ✓  SECURE: HAP bit is SET - Intel ME is disabled")

        # MMIO/SMRAM Regions & MSR Status
        if self.discoveries['mmio_regions']:
            print(f"\n━━━ MMIO/SMRAM Regions & MSR ({len(self.discoveries['mmio_regions'])}) ━━━")
            msr_entries = [m for m in self.discoveries['mmio_regions'] if m.get('type') == 'msr_status']
            mem_entries = [m for m in self.discoveries['mmio_regions'] if m.get('type') in ['memory', 'reserved_smm_candidate']]
            msr_values = [m for m in self.discoveries['mmio_regions'] if m.get('type') == 'msr_value']

            if msr_entries:
                print(f"  MSR Status:")
                for entry in msr_entries[:5]:
                    print(f"    {entry['entry']}")

            if msr_values:
                print(f"  MSR Values:")
                for entry in msr_values[:10]:
                    print(f"    {entry['entry']}")

            if mem_entries:
                print(f"  SMRAM Candidates:")
                for entry in mem_entries[:5]:
                    print(f"    {entry['entry']}")

        # WMI Interfaces
        if self.discoveries['wmi_interfaces']:
            print(f"\n━━━ WMI Interfaces ({len(self.discoveries['wmi_interfaces'])}) ━━━")
            for wmi in self.discoveries['wmi_interfaces'][:10]:
                if 'module' in wmi:
                    print(f"  Module: {wmi['module']:20} Status: {wmi.get('status', 'unknown')}")
                elif 'path' in wmi:
                    print(f"  Platform: {os.path.basename(wmi['path'])}")
                elif 'device' in wmi:
                    print(f"  Device: {wmi['device']}")

        # EFI Variables
        if self.discoveries['efi_variables']:
            print(f"\n━━━ EFI Variables ({len(self.discoveries['efi_variables'])}) ━━━")
            for efi in self.discoveries['efi_variables'][:10]:
                if 'name' in efi and 'size' in efi:
                    print(f"  {efi['name'][:60]:60} {efi.get('size', 0):6} bytes")
                elif 'note' in efi:
                    print(f"  {efi.get('name', 'Unknown')}: {efi['note']}")

        # USB Devices
        if self.discoveries['usb_devices']:
            print(f"\n━━━ Dell USB Devices ({len(self.discoveries['usb_devices'])}) ━━━")
            for usb in self.discoveries['usb_devices'][:10]:
                if 'description' in usb:
                    print(f"  {usb.get('bus_device', 'Unknown'):15} {usb['description'][:60]}")
                elif 'manufacturer' in usb:
                    print(f"  {usb['manufacturer']}: {usb.get('product', 'Unknown device')}")

        # Dell Tools
        if self.discoveries['dell_tools']:
            print(f"\n━━━ Dell Management Tools ({len(self.discoveries['dell_tools'])}) ━━━")
            tools = [t for t in self.discoveries['dell_tools'] if t.get('type') != 'package' and t.get('type') != 'service']
            packages = [t for t in self.discoveries['dell_tools'] if t.get('type') == 'package']
            services = [t for t in self.discoveries['dell_tools'] if t.get('type') == 'service']

            if tools:
                print(f"  Tools:")
                for tool in tools[:5]:
                    print(f"    {tool['name']:30} {tool.get('path', '')[:40]}")

            if packages:
                print(f"  Packages: {len(packages)} installed")
                for pkg in packages[:3]:
                    print(f"    {pkg['name']}")

            if services:
                print(f"  Services:")
                for svc in services[:5]:
                    print(f"    {svc['name']:40} {svc.get('status', 'unknown')}")

        # BIOS Settings
        if self.discoveries['bios_settings']:
            print(f"\n━━━ BIOS Settings ━━━")
            for key, value in self.discoveries['bios_settings'].items():
                print(f"  {key:20} {value}")

        # Integrated Devices
        if self.integrated_devices:
            print(f"\n━━━ Framework Integrated Devices ({len(self.integrated_devices)}) ━━━")
            for dev in self.integrated_devices:
                device_id = dev['device_id']
                name = dev['name'].replace('Device', '').strip()
                risk = dev['risk_level']
                state = dev.get('state', 'uninitialized')
                print(f"  {device_id:12} {name:30} {risk:10} {state}")

        # Summary
        print("\n" + "=" * 80)
        print("DISCOVERY SUMMARY")
        print("=" * 80)
        print("\n Basic Discovery:")
        print(f"  Kernel Modules:      {len(self.discoveries['kernel_modules'])}")
        print(f"  Device Nodes:        {len(self.discoveries['device_nodes'])}")
        print(f"  PCI Devices:         {len(self.discoveries['pci_devices'])}")
        print(f"  ACPI Tables:         {len(self.discoveries['acpi_tables'])}")
        print(f"  Running Processes:   {len(self.discoveries['processes'])}")
        print(f"  Firmware Files:      {len(self.discoveries['firmware_files'])}")
        print(f"  dmesg References:    {len(self.discoveries['dmesg_references'])}")

        print("\n Deep Hardware:")
        print(f"  MMIO/MSR Regions:    {len(self.discoveries['mmio_regions'])}")
        print(f"  Intel ME Detected:   {self.discoveries.get('intel_me', {}).get('detected', False)}")
        if self.discoveries.get('intel_me', {}).get('detected'):
            print(f"    HAP Bit:           {self.discoveries['intel_me'].get('hap_bit', 'Unknown')}")
        print(f"  WMI Interfaces:      {len(self.discoveries['wmi_interfaces'])}")
        print(f"  EFI Variables:       {len(self.discoveries['efi_variables'])}")
        print(f"  USB Devices:         {len(self.discoveries['usb_devices'])}")
        print(f"  Dell Tools:          {len(self.discoveries['dell_tools'])}")
        print(f"  BIOS Settings:       {len(self.discoveries['bios_settings'])}")

        print("\n Framework:")
        print(f"  Integrated Devices:  {len(self.integrated_devices)}")

        # DSMIL Hardware Readiness Assessment
        print("\n" + "=" * 80)
        print("DSMIL HARDWARE READINESS ASSESSMENT")
        print("=" * 80)

        # Calculate readiness score
        readiness_score = 0
        max_score = 10
        readiness_notes = []

        # Check for Dell hardware (DMI)
        dmi_entry = next((d for d in self.discoveries['sys_devices']
                        if d.get('type') == 'dmi_info'), None)
        if dmi_entry and 'data' in dmi_entry:
            vendor = dmi_entry['data'].get('sys_vendor', '')
            if 'dell' in vendor.lower():
                readiness_score += 2
                readiness_notes.append("✓ Dell hardware detected")
            else:
                readiness_notes.append("✗ Non-Dell hardware (DSMIL is Dell-specific)")
        else:
            readiness_notes.append("? Unable to determine hardware vendor")

        # Check for kernel modules
        if len(self.discoveries['kernel_modules']) > 0:
            readiness_score += 1
            readiness_notes.append(f"✓ {len(self.discoveries['kernel_modules'])} relevant kernel modules found")
        else:
            readiness_notes.append("✗ No Dell/DSMIL kernel modules detected")

        # Check for SMI interface
        smi = self.discoveries.get('smi_interface', {})
        if smi.get('port_b2_accessible') and smi.get('port_b3_accessible'):
            readiness_score += 2
            readiness_notes.append("✓ SMI interface ports (0xB2/0xB3) accessible")
        elif smi.get('dev_port_exists'):
            readiness_score += 1
            readiness_notes.append("⚠ SMI ports exist but require root access")
        else:
            readiness_notes.append("✗ SMI interface not accessible")

        # Check for WMI interfaces
        if len(self.discoveries['wmi_interfaces']) > 0:
            readiness_score += 1
            readiness_notes.append(f"✓ {len(self.discoveries['wmi_interfaces'])} WMI interfaces detected")

        # Check for Intel ME status (for MIL-SPEC)
        me = self.discoveries.get('intel_me', {})
        if me.get('detected'):
            if 'SET' in me.get('hap_bit', ''):
                readiness_score += 2
                readiness_notes.append("✓ Intel ME disabled via HAP bit (MIL-SPEC ready)")
            elif 'NOT SET' in me.get('hap_bit', ''):
                readiness_score += 1
                readiness_notes.append("⚠ Intel ME active - HAP bit not set (security concern)")
            else:
                readiness_notes.append("? Intel ME detected but HAP bit status unknown")

        # Check for Dell tools
        if len(self.discoveries['dell_tools']) > 0:
            readiness_score += 1
            readiness_notes.append(f"✓ {len(self.discoveries['dell_tools'])} Dell management tools found")

        # Check for framework integration
        if len(self.integrated_devices) >= 20:
            readiness_score += 1
            readiness_notes.append(f"✓ {len(self.integrated_devices)} devices integrated in framework")

        # Display readiness assessment
        readiness_percentage = (readiness_score / max_score) * 100
        print(f"\n  Overall Readiness: {readiness_score}/{max_score} ({readiness_percentage:.1f}%)")
        print(f"  Status: ", end='')

        if readiness_percentage >= 80:
            print("EXCELLENT - System is DSMIL-ready")
        elif readiness_percentage >= 60:
            print("GOOD - System has DSMIL capabilities")
        elif readiness_percentage >= 40:
            print("MODERATE - Limited DSMIL support")
        elif readiness_percentage >= 20:
            print("LOW - Minimal DSMIL compatibility")
        else:
            print("NONE - No DSMIL hardware detected")

        print("\n  Assessment Details:")
        for note in readiness_notes:
            print(f"    {note}")

        # Hardware compatibility analysis
        print("\n" + "=" * 80)
        print("HARDWARE COMPATIBILITY ANALYSIS")
        print("=" * 80)

        if dmi_entry and 'data' in dmi_entry:
            dmi = dmi_entry['data']
            vendor = dmi.get('sys_vendor', 'Unknown')
            product = dmi.get('product_name', 'Unknown')

            print(f"\n  System: {vendor} {product}")

            # Check for known DSMIL-compatible systems
            is_latitude = 'latitude' in product.lower()
            is_precision = 'precision' in product.lower()
            is_mil_spec = any(kw in product.lower() for kw in ['5450', '7450', 'rugged', 'mil-spec', 'atg'])

            if is_mil_spec:
                print(f"  Compatibility: ✓ CONFIRMED MIL-SPEC SYSTEM")
                print(f"  DSMIL Expected: YES - Full 108 device support expected")
            elif is_latitude or is_precision:
                print(f"  Compatibility: ⚠ POSSIBLE - Latitude/Precision detected")
                print(f"  DSMIL Expected: PARTIAL - Some devices may be present")
            else:
                print(f"  Compatibility: ✗ UNLIKELY - Not a known DSMIL platform")
                print(f"  DSMIL Expected: NO - Framework is for simulation/testing only")

        # Security recommendations
        print("\n" + "=" * 80)
        print("SECURITY RECOMMENDATIONS")
        print("=" * 80)

        recommendations_count = 0

        if me.get('detected') and 'NOT SET' in me.get('hap_bit', ''):
            recommendations_count += 1
            print(f"\n  [{recommendations_count}] CRITICAL: Intel ME is ACTIVE")
            print(f"      Issue: HAP (High Assurance Platform) bit not set")
            print(f"      Risk: Intel ME provides out-of-band management capabilities")
            print(f"      Action: For MIL-SPEC security, disable Intel ME via HAP bit")
            print(f"      Tools: Use intelmetool or BIOS settings")
        elif me.get('detected') and 'SET' in me.get('hap_bit', ''):
            print(f"\n  ✓ Intel ME is DISABLED via HAP bit (MIL-SPEC compliant)")

        if not self.check_root_privileges():
            recommendations_count += 1
            print(f"\n  [{recommendations_count}] INFO: Limited privilege access")
            print(f"      Issue: Not running as root")
            print(f"      Impact: Cannot read MSR, MMIO regions, or detailed ME status")
            print(f"      Action: Run with sudo for complete hardware discovery")
            print(f"      Command: sudo python3 dsmil_discover.py")

        if smi.get('smi_disabled'):
            recommendations_count += 1
            print(f"\n  [{recommendations_count}] WARNING: SMI disabled in kernel")
            print(f"      Issue: 'nosmi' kernel parameter detected")
            print(f"      Impact: DSMIL SMI interface will not function")
            print(f"      Action: Remove 'nosmi' from kernel command line if DSMIL access needed")

        if len(self.discoveries['kernel_modules']) == 0:
            recommendations_count += 1
            print(f"\n  [{recommendations_count}] INFO: No Dell kernel modules loaded")
            print(f"      Modules: dcdbas, dell_wmi, dell_smbios, dell_smm_hwmon")
            print(f"      Action: Install and load Dell kernel modules for hardware access")
            print(f"      Command: modprobe dcdbas dell_wmi dell_smbios")

        if me.get('amt_enabled') == 'YES (Ports 16992/16993 listening)':
            recommendations_count += 1
            print(f"\n  [{recommendations_count}] WARNING: Intel AMT is ACTIVE")
            print(f"      Issue: Active Management Technology detected")
            print(f"      Risk: Remote management interface is enabled")
            print(f"      Action: Disable AMT in BIOS if not required")

        if recommendations_count == 0:
            print("\n  ✓ No security issues detected")
            print("  ✓ System appears to be configured correctly")

        # Deployment recommendations
        print("\n" + "=" * 80)
        print("DSMIL DEPLOYMENT RECOMMENDATIONS")
        print("=" * 80)

        print("\n  Framework Status:")
        print(f"    • Integrated Devices: {len(self.integrated_devices)}/108 (20.4%)")
        print(f"    • Framework Version: {self.framework_status.get('version', 'Unknown')}")

        print("\n  Next Steps for DSMIL Development:")
        if readiness_percentage >= 60:
            print("    1. ✓ Hardware is DSMIL-capable")
            print("    2. → Test device enumeration with dsmil_probe.py")
            print("    3. → Use interactive menu: python3 dsmil_menu.py")
            print("    4. → Monitor devices with dsmil_monitor.py")
            print("    5. → Integrate additional devices from COMPLETE_DEVICE_DISCOVERY.md")
        else:
            print("    1. ⚠ Hardware may not support DSMIL")
            print("    2. → Framework can be used for simulation/testing")
            print("    3. → Review device documentation in 02-tools/dsmil-devices/")
            print("    4. → Use on actual Dell Latitude/Precision MIL-SPEC system for real access")

        print("\n  Tools Available:")
        print("    • dsmil_discover.py  - This hardware discovery tool")
        print("    • dsmil_integration.py - Device registration framework")
        print("    • dsmil_menu.py      - Interactive device control menu")
        print("    • dsmil_probe.py     - Device functional testing")
        print("    • Location: ./02-tools/dsmil-devices/")

        # Quick reference
        print("\n" + "=" * 80)
        print("QUICK REFERENCE")
        print("=" * 80)

        print("\n  Discovery Commands:")
        print("    Basic scan:      python3 dsmil_discover.py")
        print("    Deep scan:       sudo python3 dsmil_discover.py")
        print("    Save report:     sudo python3 dsmil_discover.py > report.txt")
        print("    Shortcut:        ./dsmil-discover.sh --sudo --output report.txt")

        print("\n  Framework Commands:")
        print("    Interactive menu: python3 02-tools/dsmil-devices/dsmil_menu.py")
        print("    Device probe:     python3 02-tools/dsmil-devices/dsmil_probe.py")
        print("    List devices:     python3 -c 'from dsmil_auto_discover import list_devices; list_devices()'")

        print("\n  Intel ME Commands:")
        print("    Check HAP bit:    sudo intelmetool -m")
        print("    ME information:   sudo intelmetool -s")
        print("    Disable ME:       Set HAP bit in BIOS/UEFI settings")

        print("\n  Kernel Module Commands:")
        print("    Load MSR module:  sudo modprobe msr")
        print("    Load Dell modules: sudo modprobe dcdbas dell_wmi dell_smbios")
        print("    Check modules:    lsmod | grep -E 'dell|smi|mei'")

        print("\n  Documentation:")
        print("    Device catalog:   02-tools/dsmil-devices/COMPLETE_DEVICE_DISCOVERY.md")
        print("    Integration guide: 02-tools/dsmil-devices/README.md")
        print("    Framework summary: 02-tools/dsmil-devices/INTEGRATION_SUMMARY.md")

        print("=" * 80 + "\n")


def main():
    """Main entry point"""
    print("\n╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "DSMIL LOCAL SYSTEM DISCOVERY" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")

    discovery = DSMILDiscovery()
    discovery.run_discovery()
    discovery.print_report()


if __name__ == "__main__":
    main()
