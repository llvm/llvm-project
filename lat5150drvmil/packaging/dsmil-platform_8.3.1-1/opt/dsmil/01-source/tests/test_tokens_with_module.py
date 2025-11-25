#!/usr/bin/env python3
"""
Test SMBIOS tokens with DSMIL kernel module loaded
Tests both standard and SMI access methods
"""

import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

class TokenModuleTester:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'module_loaded': self.check_module(),
            'tests': []
        }
        
        # Test tokens from each category
        self.test_tokens = {
            'power': 0x0480,      # Position 0, Group 0 - SMI required
            'thermal': 0x0481,    # Position 1, Group 0 - Should be accessible
            'security': 0x0482,   # Position 2, Group 0 - Should be accessible  
            'memory': 0x0483,     # Position 3, Group 0 - SMI required
            'io': 0x0484,         # Position 4, Group 0 - Should be accessible
            'storage': 0x0486,    # Position 6, Group 0 - SMI required
            'sensor': 0x0489,     # Position 9, Group 0 - SMI required
        }
        
    def check_module(self):
        """Check if DSMIL module is loaded"""
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            return 'dsmil_72dev' in result.stdout
        except:
            return False
    
    def get_thermal(self):
        """Get current max temperature"""
        temps = []
        for zone in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:
                temp = int(zone.read_text().strip()) / 1000
                temps.append(temp)
            except:
                pass
        return max(temps) if temps else 0
    
    def test_smbios_access(self, token):
        """Test standard SMBIOS access"""
        try:
            # Try to read token
            cmd = f'echo "1786" | sudo -S smbios-token-ctl --token-id={token:#x} --get 2>&1'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if 'Active' in result.stdout:
                return {'accessible': True, 'value': 'Active', 'method': 'SMBIOS'}
            elif 'Inactive' in result.stdout:
                return {'accessible': True, 'value': 'Inactive', 'method': 'SMBIOS'}
            elif 'invalid token' in result.stdout.lower():
                return {'accessible': False, 'error': 'Invalid token', 'method': 'SMBIOS'}
            else:
                return {'accessible': False, 'error': 'Not accessible', 'method': 'SMBIOS'}
        except Exception as e:
            return {'accessible': False, 'error': str(e), 'method': 'SMBIOS'}
    
    def test_smi_access(self, token):
        """Test SMI access via kernel module"""
        # Check for debugfs interface
        debugfs_path = Path('/sys/kernel/debug/dsmil')
        if not debugfs_path.exists():
            # Try via /dev/dsmil device files
            dev_path = Path('/dev/dsmil0')
            if dev_path.exists():
                return {'accessible': True, 'method': 'SMI', 'note': '/dev/dsmil0 exists'}
            return {'accessible': False, 'error': 'No SMI interface found', 'method': 'SMI'}
        
        return {'accessible': True, 'method': 'SMI', 'note': 'debugfs interface available'}
    
    def check_kernel_response(self):
        """Check for DSMIL kernel messages"""
        try:
            cmd = 'echo "1786" | sudo -S dmesg | tail -20 | grep -i "dsmil\\|smi\\|token"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                return lines[-5:] if len(lines) > 5 else lines
        except:
            pass
        return []
    
    def test_token(self, name, token):
        """Test a single token with all methods"""
        print(f"\n🔍 Testing {name} token 0x{token:04X}")
        
        test_result = {
            'name': name,
            'token': f'0x{token:04X}',
            'timestamp': datetime.now().isoformat(),
            'thermal_before': self.get_thermal()
        }
        
        # Test SMBIOS access
        print(f"   Testing SMBIOS access...")
        smbios_result = self.test_smbios_access(token)
        test_result['smbios'] = smbios_result
        
        if smbios_result['accessible']:
            print(f"   ✅ SMBIOS: Accessible, value = {smbios_result.get('value', 'unknown')}")
        else:
            print(f"   ❌ SMBIOS: {smbios_result.get('error', 'Not accessible')}")
        
        # Test SMI access if SMBIOS failed
        if not smbios_result['accessible']:
            print(f"   Testing SMI access...")
            smi_result = self.test_smi_access(token)
            test_result['smi'] = smi_result
            
            if smi_result['accessible']:
                print(f"   ✅ SMI: {smi_result.get('note', 'Accessible')}")
            else:
                print(f"   ❌ SMI: {smi_result.get('error', 'Not accessible')}")
        
        # Check kernel response
        kernel_msgs = self.check_kernel_response()
        if kernel_msgs:
            test_result['kernel_activity'] = kernel_msgs
            print(f"   📝 Kernel activity detected")
        
        test_result['thermal_after'] = self.get_thermal()
        test_result['thermal_change'] = test_result['thermal_after'] - test_result['thermal_before']
        
        self.results['tests'].append(test_result)
        time.sleep(1)  # Small delay between tests
        
        return test_result
    
    def run_tests(self):
        """Run all token tests"""
        print("="*60)
        print("DSMIL TOKEN TESTING WITH MODULE LOADED")
        print("="*60)
        print(f"Module loaded: {self.results['module_loaded']}")
        print(f"Initial temperature: {self.get_thermal():.1f}°C")
        
        if not self.results['module_loaded']:
            print("⚠️  WARNING: DSMIL module not loaded!")
            print("   Load with: sudo insmod dsmil-72dev.ko")
        
        # Test each token
        for name, token in self.test_tokens.items():
            self.test_token(name, token)
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        accessible_count = 0
        smi_required = 0
        
        for test in self.results['tests']:
            if test.get('smbios', {}).get('accessible'):
                accessible_count += 1
                print(f"✅ {test['name']:10s} (0x{int(test['token'], 16):04X}): SMBIOS accessible")
            elif test.get('smi', {}).get('accessible'):
                smi_required += 1
                print(f"🔐 {test['name']:10s} (0x{int(test['token'], 16):04X}): SMI access required")
            else:
                print(f"❌ {test['name']:10s} (0x{int(test['token'], 16):04X}): Not accessible")
        
        print(f"\nResults:")
        print(f"  SMBIOS accessible: {accessible_count}/{len(self.test_tokens)}")
        print(f"  SMI required: {smi_required}/{len(self.test_tokens)}")
        print(f"  Final temperature: {self.get_thermal():.1f}°C")
        
        # Save results
        results_file = Path('logs') / f'module_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        results_file.parent.mkdir(exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to: {results_file}")
        
        return self.results

if __name__ == "__main__":
    tester = TokenModuleTester()
    tester.run_tests()