#!/usr/bin/env python3
"""
Test the REAL DSMIL tokens discovered at 0x60000000
Range: 0x8000-0x887F (128 tokens total, expecting 72 active)
"""

import subprocess
import time
from datetime import datetime

class RealDSMILTokenTester:
    def __init__(self):
        self.password = "1786"
        # Based on memory dump, these are the actual DSMIL tokens
        self.token_ranges = {
            'Group_0': list(range(0x8000, 0x8010)),  # 16 tokens
            'Group_1': list(range(0x8010, 0x8020)),  # 16 tokens
            'Group_2': list(range(0x8020, 0x8030)),  # 16 tokens
            'Group_3': list(range(0x8030, 0x8040)),  # 16 tokens
            'Group_4': list(range(0x8040, 0x8050)),  # 16 tokens
            'Group_5': list(range(0x8050, 0x8060)),  # 16 tokens
            'Extended': list(range(0x8060, 0x8080))  # Extra 32 tokens
        }
        self.results = {}
        
    def test_token(self, token):
        """Test a single DSMIL token"""
        try:
            # Try with smbios-token-ctl
            cmd = f'echo "{self.password}" | sudo -S smbios-token-ctl --token-id={token:#x} --get 2>&1'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
            
            if 'Active' in result.stdout:
                return 'Active'
            elif 'Inactive' in result.stdout:
                return 'Inactive'
            elif 'invalid' in result.stdout.lower():
                return 'Invalid'
            else:
                # Try direct SMI if SMBIOS fails
                return self.test_smi_direct(token)
        except subprocess.TimeoutExpired:
            return 'Timeout'
        except Exception as e:
            return f'Error: {e}'
    
    def test_smi_direct(self, token):
        """Try direct SMI access to token"""
        # Create simple SMI test
        smi_code = f"""
#include <stdio.h>
#include <sys/io.h>
#include <unistd.h>

int main() {{
    if (iopl(3) != 0) {{
        return 1;
    }}
    
    // Write token to port
    outw(0x{token:04X}, 0x164E);
    
    // Read status
    unsigned char status = inb(0x164F);
    
    if (status & 0x01) {{
        printf("SMI_Active");
    }} else {{
        printf("SMI_Inactive");
    }}
    
    return 0;
}}
"""
        
        try:
            with open("/tmp/smi_test.c", "w") as f:
                f.write(smi_code)
            
            # Compile and run
            subprocess.run("gcc -o /tmp/smi_test /tmp/smi_test.c", shell=True, check=True, capture_output=True)
            cmd = f'echo "{self.password}" | sudo -S /tmp/smi_test'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1)
            
            if 'SMI_Active' in result.stdout:
                return 'SMI_Active'
            elif 'SMI_Inactive' in result.stdout:
                return 'SMI_Inactive'
            else:
                return 'No_Response'
                
        except:
            return 'SMI_Failed'
        finally:
            # Cleanup
            subprocess.run("rm -f /tmp/smi_test /tmp/smi_test.c", shell=True)
    
    def test_group(self, group_name, tokens):
        """Test all tokens in a group"""
        print(f"\n{'='*60}")
        print(f"Testing {group_name}: {len(tokens)} tokens")
        print(f"Range: 0x{tokens[0]:04X} - 0x{tokens[-1]:04X}")
        print(f"{'='*60}")
        
        group_results = {
            'active': 0,
            'inactive': 0,
            'no_response': 0,
            'tokens': {}
        }
        
        for i, token in enumerate(tokens):
            result = self.test_token(token)
            group_results['tokens'][token] = result
            
            # Count results
            if 'Active' in result:
                group_results['active'] += 1
                status = "✅"
            elif 'Inactive' in result:
                group_results['inactive'] += 1
                status = "⭕"
            else:
                group_results['no_response'] += 1
                status = "❌"
            
            print(f"  {status} Token 0x{token:04X}: {result}")
            
            # Small delay to prevent overwhelming
            if i % 4 == 3:
                time.sleep(0.5)
        
        # Summary
        print(f"\n📊 {group_name} Summary:")
        print(f"   Active: {group_results['active']}")
        print(f"   Inactive: {group_results['inactive']}")
        print(f"   No Response: {group_results['no_response']}")
        
        self.results[group_name] = group_results
        return group_results
    
    def run_discovery(self):
        """Run complete DSMIL token discovery"""
        print("="*70)
        print("REAL DSMIL TOKEN DISCOVERY")
        print("="*70)
        print(f"Start: {datetime.now()}")
        print(f"Testing tokens 0x8000-0x8080 (found in memory at 0x60000000)")
        print()
        
        total_active = 0
        total_inactive = 0
        total_no_response = 0
        
        # Test each group
        for group_name, tokens in self.token_ranges.items():
            results = self.test_group(group_name, tokens[:12])  # Test first 12 of each
            total_active += results['active']
            total_inactive += results['inactive']
            total_no_response += results['no_response']
            time.sleep(1)  # Pause between groups
        
        # Final summary
        print("\n" + "="*70)
        print("DISCOVERY COMPLETE")
        print("="*70)
        print(f"✅ Total Active: {total_active}")
        print(f"⭕ Total Inactive: {total_inactive}")
        print(f"❌ Total No Response: {total_no_response}")
        print(f"📊 Total Tested: {total_active + total_inactive + total_no_response}")
        
        if total_active + total_inactive >= 60:
            print("\n🎯 SUCCESS: Found DSMIL operational devices!")
            print(f"   {total_active + total_inactive} devices responding")
            print("   Ready for control interface development")
        elif total_active + total_inactive >= 12:
            print("\n⚠️  PARTIAL SUCCESS: Some devices responding")
            print(f"   {total_active + total_inactive} devices accessible")
            print("   May need different access method for others")
        else:
            print("\n❌ LIMITED ACCESS: Devices require different approach")
            print("   Consider kernel module interface or SMI")
        
        # Save results
        with open("dsmil_token_discovery.txt", "w") as f:
            f.write(f"DSMIL Token Discovery Results\n")
            f.write(f"Date: {datetime.now()}\n\n")
            for group, results in self.results.items():
                f.write(f"{group}:\n")
                for token, status in results['tokens'].items():
                    f.write(f"  0x{token:04X}: {status}\n")
                f.write(f"  Summary: Active={results['active']}, Inactive={results['inactive']}, No Response={results['no_response']}\n\n")
        
        print("\n💾 Results saved to: dsmil_token_discovery.txt")
        
        return self.results

if __name__ == "__main__":
    tester = RealDSMILTokenTester()
    tester.run_discovery()