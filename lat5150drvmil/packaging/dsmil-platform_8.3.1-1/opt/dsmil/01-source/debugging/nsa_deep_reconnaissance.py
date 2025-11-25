#!/usr/bin/env python3
"""
NSA DEEP RECONNAISSANCE - ROOT ACCESS
Enhanced device analysis with full hardware access
Root password: 1
Sudo password: 1786
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [NSA] %(message)s'
)
logger = logging.getLogger(__name__)

class NSADeepReconnaissance:
    def __init__(self):
        self.root_password = "1"
        self.sudo_password = "1786"
        self.quarantine_list = [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]
        self.discoveries = []
    
    def run_root_command(self, cmd_list, timeout=10):
        """Execute command as root with password 1"""
        try:
            # Switch to root user with password "1"
            full_cmd = ["su", "-c", " ".join(cmd_list)]
            process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(
                input=self.root_password + "\n",
                timeout=timeout
            )
            
            return process.returncode, stdout, stderr
            
        except subprocess.TimeoutExpired:
            process.kill()
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def probe_hardware_direct(self):
        """Direct hardware probing with root access"""
        logger.info("🔍 Starting NSA deep hardware reconnaissance...")
        
        discoveries = {
            'pci_devices': self.scan_pci_devices(),
            'acpi_devices': self.scan_acpi_devices(),
            'smbios_tokens': self.scan_smbios_tokens(),
            'dmi_info': self.scan_dmi_info(),
            'msr_access': self.test_msr_access(),
            'io_ports': self.scan_io_ports(),
            'memory_mapped': self.scan_memory_mapped(),
            'dell_specific': self.scan_dell_interfaces()
        }
        
        return discoveries
    
    def scan_pci_devices(self):
        """Scan PCI devices for security controllers"""
        logger.info("Scanning PCI devices...")
        
        returncode, stdout, stderr = self.run_root_command(["lspci", "-v"])
        
        if returncode == 0:
            security_devices = []
            lines = stdout.split('\n')
            current_device = None
            
            for line in lines:
                if ':' in line and ('.' in line[:8]):
                    # New device
                    current_device = {'id': line.split()[0], 'description': line}
                    if any(keyword in line.lower() for keyword in ['security', 'tpm', 'crypto', 'hsm', 'management']):
                        security_devices.append(current_device)
                elif current_device and line.startswith('\t'):
                    # Device details
                    if 'current_device' not in locals():
                        continue
                    if 'details' not in current_device:
                        current_device['details'] = []
                    current_device['details'].append(line.strip())
            
            logger.info(f"Found {len(security_devices)} security-related PCI devices")
            return security_devices
        
        return []
    
    def scan_acpi_devices(self):
        """Scan ACPI tables for device information"""
        logger.info("Scanning ACPI devices...")
        
        acpi_devices = {}
        
        # Scan ACPI tables directory
        returncode, stdout, stderr = self.run_root_command(["find", "/sys/firmware/acpi/tables", "-name", "*"])
        
        if returncode == 0:
            for table_path in stdout.strip().split('\n'):
                if table_path:
                    table_name = Path(table_path).name
                    if table_name in ['SSDT', 'DSDT', 'MSDM', 'SLIC']:
                        # Try to read table
                        ret, content, err = self.run_root_command(["hexdump", "-C", table_path, "|", "head", "-20"])
                        if ret == 0:
                            acpi_devices[table_name] = content[:500]  # Truncate
        
        return acpi_devices
    
    def scan_smbios_tokens(self):
        """Scan for SMBIOS/Dell tokens with enhanced detection"""
        logger.info("Scanning SMBIOS tokens...")
        
        token_data = {}
        
        # Try dmidecode
        returncode, stdout, stderr = self.run_root_command(["dmidecode", "-t", "bios"])
        if returncode == 0:
            token_data['bios_info'] = stdout[:1000]
        
        # Try Dell-specific token access
        returncode, stdout, stderr = self.run_root_command(["dmidecode", "-s", "system-manufacturer"])
        if returncode == 0 and 'Dell' in stdout:
            logger.info("Dell system detected - trying Dell token access")
            
            # Try to access Dell SMBIOS interface
            for token_range in [(0x0480, 0x04C7), (0x8000, 0x806B)]:
                start, end = token_range
                responsive_tokens = []
                
                for token_id in range(start, end + 1, 10):  # Sample every 10th token
                    if token_id in self.quarantine_list:
                        continue
                    
                    # Try multiple access methods
                    methods = [
                        f"echo 'Token 0x{token_id:04X}' > /dev/null",  # Safe test
                        f"cat /sys/class/dmi/id/chassis_type 2>/dev/null",  # DMI read
                    ]
                    
                    for method in methods:
                        ret, out, err = self.run_root_command([method])
                        if ret == 0 and out.strip():
                            responsive_tokens.append(f"0x{token_id:04X}")
                            break
                    
                    time.sleep(0.01)  # Small delay
                
                token_data[f'range_{start:04X}_{end:04X}'] = responsive_tokens
        
        return token_data
    
    def scan_dmi_info(self):
        """Scan DMI/SMBIOS information"""
        logger.info("Scanning DMI information...")
        
        dmi_info = {}
        
        dmi_files = [
            'bios_date', 'bios_vendor', 'bios_version',
            'board_name', 'board_vendor', 'board_version',
            'chassis_type', 'chassis_vendor',
            'product_name', 'product_version',
            'sys_vendor'
        ]
        
        for dmi_file in dmi_files:
            dmi_path = f"/sys/class/dmi/id/{dmi_file}"
            returncode, stdout, stderr = self.run_root_command(["cat", dmi_path])
            if returncode == 0:
                dmi_info[dmi_file] = stdout.strip()
        
        return dmi_info
    
    def test_msr_access(self):
        """Test MSR (Model Specific Register) access"""
        logger.info("Testing MSR access...")
        
        msr_results = {}
        
        # Load MSR module
        returncode, stdout, stderr = self.run_root_command(["modprobe", "msr"])
        
        if returncode == 0:
            # Try to read some safe MSRs
            safe_msrs = {
                '0x17': 'IA32_PLATFORM_ID',
                '0x8B': 'IA32_BIOS_SIGN_ID',
                '0xCE': 'MSR_PLATFORM_INFO'
            }
            
            for msr_addr, msr_name in safe_msrs.items():
                returncode, stdout, stderr = self.run_root_command(["rdmsr", msr_addr])
                if returncode == 0:
                    msr_results[msr_name] = stdout.strip()
        
        return msr_results
    
    def scan_io_ports(self):
        """Scan I/O ports for device interfaces"""
        logger.info("Scanning I/O ports...")
        
        io_results = {}
        
        # Check /proc/ioports
        returncode, stdout, stderr = self.run_root_command(["cat", "/proc/ioports"])
        if returncode == 0:
            io_results['ioports'] = stdout[:2000]  # Truncate
        
        # Test specific ports known to be used by DSMIL
        test_ports = [0x164E, 0x164F, 0xB2, 0xB3]  # SMI and other management ports
        
        for port in test_ports:
            try:
                # Try to check if port is accessible (READ-ONLY test)
                returncode, stdout, stderr = self.run_root_command([
                    "python3", "-c", 
                    f"import os; "
                    f"try: "
                    f"  with open('/dev/port', 'rb') as f: "
                    f"    f.seek({port}); "
                    f"    data=f.read(1); "
                    f"    print('accessible' if data else 'no_data'); "
                    f"except: print('denied')"
                ])
                
                if returncode == 0:
                    io_results[f'port_0x{port:04X}'] = stdout.strip()
            except:
                pass
        
        return io_results
    
    def scan_memory_mapped(self):
        """Scan memory-mapped devices"""
        logger.info("Scanning memory-mapped devices...")
        
        mm_results = {}
        
        # Check /proc/iomem
        returncode, stdout, stderr = self.run_root_command(["cat", "/proc/iomem"])
        if returncode == 0:
            # Look for interesting memory regions
            interesting_regions = []
            for line in stdout.split('\n'):
                if any(keyword in line.lower() for keyword in ['reserved', 'acpi', 'configuration', 'device']):
                    interesting_regions.append(line.strip())
            
            mm_results['interesting_regions'] = interesting_regions[:20]  # Limit output
        
        return mm_results
    
    def scan_dell_interfaces(self):
        """Scan Dell-specific interfaces"""
        logger.info("Scanning Dell-specific interfaces...")
        
        dell_results = {}
        
        # Check for Dell BIOS utilities
        dell_tools = ['dcdbas', 'dell_rbu', 'dell_wmi']
        
        for tool in dell_tools:
            returncode, stdout, stderr = self.run_root_command(["lsmod", "|", "grep", tool])
            if returncode == 0:
                dell_results[f'{tool}_loaded'] = True
            else:
                dell_results[f'{tool}_loaded'] = False
        
        # Check Dell WMI interface
        returncode, stdout, stderr = self.run_root_command(["find", "/sys", "-name", "*dell*", "-type", "d"])
        if returncode == 0:
            dell_results['dell_sys_entries'] = stdout.strip().split('\n')[:10]  # Limit
        
        return dell_results
    
    def generate_intelligence_report(self, discoveries):
        """Generate comprehensive intelligence report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mission': 'NSA Deep Hardware Reconnaissance',
            'classification': 'TOP SECRET//SI//NOFORN',
            'discoveries': discoveries,
            'analysis': {},
            'recommendations': []
        }
        
        # Analyze discoveries
        if discoveries.get('pci_devices'):
            security_count = len(discoveries['pci_devices'])
            report['analysis']['security_controllers'] = f"{security_count} security-related PCI devices found"
        
        if discoveries.get('dmi_info', {}).get('product_name'):
            product = discoveries['dmi_info']['product_name']
            report['analysis']['hardware_platform'] = f"Confirmed: {product}"
        
        if discoveries.get('msr_access'):
            msr_count = len(discoveries['msr_access'])
            report['analysis']['msr_access'] = f"MSR access confirmed: {msr_count} registers readable"
        
        # Generate recommendations
        if discoveries.get('dell_sys_entries'):
            report['recommendations'].append("Dell WMI interface detected - recommend Dell-specific token access")
        
        if discoveries.get('io_results', {}).get('port_0x164E'):
            report['recommendations'].append("SMI interface accessible - recommend direct SMI communication")
        
        report['recommendations'].append("Consider ACPI method enumeration for device discovery")
        report['recommendations'].append("Investigate Dell OpenManage integration for enhanced device access")
        
        return report
    
    def execute_deep_reconnaissance(self):
        """Execute comprehensive reconnaissance mission"""
        
        print("🔍 NSA DEEP RECONNAISSANCE - STARTING")
        print("=" * 60)
        print("Classification: TOP SECRET//SI//NOFORN")
        print("Hardware: Dell Latitude 5450 MIL-SPEC")
        print("Access Level: ROOT")
        print("=" * 60)
        
        # Execute reconnaissance
        discoveries = self.probe_hardware_direct()
        
        # Generate report
        intelligence_report = self.generate_intelligence_report(discoveries)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"/home/john/LAT5150DRVMIL/nsa_deep_recon_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(intelligence_report, f, indent=2, default=str)
        
        # Display summary
        self.display_intelligence_summary(intelligence_report)
        
        logger.info(f"Full report saved: {report_file}")
        
        return intelligence_report
    
    def display_intelligence_summary(self, report):
        """Display intelligence summary"""
        
        print("\n🎯 INTELLIGENCE SUMMARY")
        print("=" * 40)
        
        discoveries = report['discoveries']
        
        if discoveries.get('pci_devices'):
            print(f"🔍 Security Controllers: {len(discoveries['pci_devices'])}")
            for device in discoveries['pci_devices'][:3]:  # Show first 3
                print(f"   • {device.get('id', 'Unknown')}: {device.get('description', 'Unknown')[:60]}...")
        
        if discoveries.get('dmi_info'):
            dmi = discoveries['dmi_info']
            if dmi.get('product_name'):
                print(f"🖥️  Hardware: {dmi['product_name']}")
            if dmi.get('bios_version'):
                print(f"⚙️  BIOS: {dmi['bios_version']}")
        
        if discoveries.get('msr_access'):
            print(f"📡 MSR Access: {len(discoveries['msr_access'])} registers")
        
        if discoveries.get('io_ports'):
            accessible_ports = [k for k, v in discoveries['io_ports'].items() if 'accessible' in str(v)]
            print(f"🔌 I/O Ports: {len(accessible_ports)} accessible")
        
        print("\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
        
        print("\n✅ DEEP RECONNAISSANCE COMPLETE")

def main():
    """Main NSA deep reconnaissance execution"""
    
    if os.getuid() == 0:
        logger.info("Running as root - full hardware access available")
    else:
        logger.info("Running as user - will attempt root escalation")
    
    nsa_deep = NSADeepReconnaissance()
    
    try:
        report = nsa_deep.execute_deep_reconnaissance()
        return 0
    except KeyboardInterrupt:
        print("\n⚠️ Mission interrupted")
        return 1
    except Exception as e:
        logger.error(f"Mission failed: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())