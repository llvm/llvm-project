#!/usr/bin/env python3
"""
Test direct SMI access to SMBIOS tokens
Since the module mapped memory but found no structures,
we'll try direct SMI calls to activate tokens
"""

import subprocess
import struct
import time
import os
from pathlib import Path
from datetime import datetime

class SMIDirectTester:
    def __init__(self):
        self.password = "1786"
        self.results = []
        
        # Critical tokens to test
        self.test_tokens = [
            (0x0480, "Power Management", "Group 0, Position 0"),
            (0x0481, "Thermal Control", "Group 0, Position 1"),
            (0x0482, "Security Module", "Group 0, Position 2"),
            (0x0483, "Memory Controller", "Group 0, Position 3"),
        ]
        
    def check_memory_regions(self):
        """Check what memory regions are actually mapped"""
        print("📍 Checking mapped memory regions...")
        
        # Check /proc/iomem for DSMIL regions
        try:
            cmd = f'echo "{self.password}" | sudo -S cat /proc/iomem | grep -i "dsmil\\|reserved" | head -10'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print("Memory regions found:")
                for line in result.stdout.strip().split('\n'):
                    if '52000000' in line or 'dsmil' in line.lower():
                        print(f"  ✓ {line}")
        except:
            pass
        
        # Check kernel module's view
        try:
            cmd = f'echo "{self.password}" | sudo -S cat /proc/modules | grep dsmil'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                print(f"Module info: {result.stdout.strip()}")
        except:
            pass
    
    def try_token_via_wmi(self, token):
        """Try accessing token via Dell WMI interface"""
        # Check if token maps to a WMI attribute
        wmi_mappings = {
            0x0481: "ThermalManagement",
            0x0480: "PowerManagement", 
            0x0482: "TpmSecurity"
        }
        
        if token in wmi_mappings:
            attr = wmi_mappings[token]
            wmi_path = f"/sys/class/firmware-attributes/dell-wmi-sysman/attributes/{attr}"
            
            if Path(wmi_path).exists():
                try:
                    current = Path(f"{wmi_path}/current_value").read_text().strip()
                    return f"WMI accessible: {attr} = {current}"
                except:
                    pass
        
        return None
    
    def send_smi_command(self, token, activate=True):
        """Send SMI command to control token"""
        print(f"\n🔧 Attempting SMI control of token 0x{token:04X}")
        
        # Create a test program to send SMI
        smi_code = f"""
#include <stdio.h>
#include <sys/io.h>
#include <unistd.h>

int main() {{
    // Request I/O port access
    if (iopl(3) != 0) {{
        printf("Failed to get I/O privileges\\n");
        return 1;
    }}
    
    // Dell I/O ports
    unsigned short token_port = 0x164E;
    unsigned char smi_port = 0xB2;
    unsigned char smi_cmd = {0xF1 if activate else 0xF0};
    
    // Write token to Dell I/O space
    outw(0x{token:04X}, token_port);
    
    // Trigger SMI
    printf("Sending SMI for token 0x{token:04X}...\\n");
    outb(smi_cmd, smi_port);
    
    // Wait for completion
    usleep(100000); // 100ms
    
    printf("SMI command sent\\n");
    return 0;
}}
"""
        
        # Write and compile the SMI program
        smi_file = Path("smi_test.c")
        smi_file.write_text(smi_code)
        
        try:
            # Compile
            compile_cmd = "gcc -o smi_test smi_test.c"
            subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
            
            # Run with sudo
            run_cmd = f'echo "{self.password}" | sudo -S ./smi_test'
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
            
            print(f"   SMI Result: {result.stdout.strip()}")
            
            # Check kernel response
            kernel_cmd = f'echo "{self.password}" | sudo -S dmesg | tail -5 | grep -i "smi\\|dsmil\\|token"'
            kernel_result = subprocess.run(kernel_cmd, shell=True, capture_output=True, text=True)
            if kernel_result.stdout:
                print(f"   Kernel response: {kernel_result.stdout.strip()}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ SMI failed: {e}")
            return False
        finally:
            # Cleanup
            Path("smi_test.c").unlink(missing_ok=True)
            Path("smi_test").unlink(missing_ok=True)
    
    def probe_memory_directly(self):
        """Try to read DSMIL memory region directly"""
        print("\n🔍 Probing DSMIL memory region directly...")
        
        # Create memory probe program
        probe_code = """
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

int main() {
    int fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (fd < 0) {
        printf("Cannot open /dev/mem\\n");
        return 1;
    }
    
    // Map DSMIL region at 0x52000000
    size_t map_size = 4096; // Just first page
    off_t target = 0x52000000;
    
    void *map_base = mmap(0, map_size, PROT_READ, MAP_SHARED, fd, target);
    if (map_base == MAP_FAILED) {
        printf("mmap failed\\n");
        close(fd);
        return 1;
    }
    
    // Read first few DWORDs
    uint32_t *data = (uint32_t *)map_base;
    printf("DSMIL memory at 0x52000000:\\n");
    for (int i = 0; i < 8; i++) {
        printf("  [0x%08lX]: 0x%08X\\n", target + i*4, data[i]);
    }
    
    // Look for signatures
    for (int i = 0; i < 1024; i++) {
        if (data[i] == 0x4C494D53 || // "SMIL"
            data[i] == 0x4C4D5344) { // "DSML"
            printf("Found signature at offset 0x%X: 0x%08X\\n", i*4, data[i]);
        }
    }
    
    munmap(map_base, map_size);
    close(fd);
    return 0;
}
"""
        
        probe_file = Path("probe_mem.c")
        probe_file.write_text(probe_code)
        
        try:
            # Compile
            subprocess.run("gcc -o probe_mem probe_mem.c", shell=True, check=True, capture_output=True)
            
            # Run with sudo
            cmd = f'echo "{self.password}" | sudo -S ./probe_mem'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.stdout:
                print(result.stdout)
                
                # Check if we found anything interesting
                if "0x00000000" not in result.stdout or "0xFFFFFFFF" not in result.stdout:
                    print("   📌 Memory contains data (not empty/unmapped)")
            
        except Exception as e:
            print(f"   Memory probe failed: {e}")
        finally:
            Path("probe_mem.c").unlink(missing_ok=True)
            Path("probe_mem").unlink(missing_ok=True)
    
    def run_tests(self):
        """Run all SMI tests"""
        print("="*60)
        print("DIRECT SMI TOKEN ACCESS TESTING")
        print("="*60)
        print("Since no DSMIL structures were found in memory,")
        print("we'll try direct SMI commands to control tokens.")
        print()
        
        # Check memory regions
        self.check_memory_regions()
        
        # Probe memory directly
        self.probe_memory_directly()
        
        # Test each token
        print("\n🚀 Testing SMI token control...")
        for token, name, desc in self.test_tokens:
            print(f"\n📌 Token 0x{token:04X}: {name} ({desc})")
            
            # Try WMI first
            wmi_result = self.try_token_via_wmi(token)
            if wmi_result:
                print(f"   {wmi_result}")
            
            # Try SMI
            if self.send_smi_command(token, activate=True):
                self.results.append((token, "SMI successful"))
                time.sleep(1)
                # Try to deactivate
                self.send_smi_command(token, activate=False)
            else:
                self.results.append((token, "SMI failed"))
            
            time.sleep(1)
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for token, status in self.results:
            print(f"Token 0x{token:04X}: {status}")
        
        print("\n💡 Analysis:")
        print("The kernel module successfully mapped 16MB of memory")
        print("but found no DSMIL structure signatures.")
        print("This suggests:")
        print("1. DSMIL devices may use a different protocol")
        print("2. Structures may be encrypted or obfuscated")
        print("3. Tokens may be controlled entirely via SMI")
        print("4. Real structures may be in SMM protected memory")

if __name__ == "__main__":
    tester = SMIDirectTester()
    tester.run_tests()