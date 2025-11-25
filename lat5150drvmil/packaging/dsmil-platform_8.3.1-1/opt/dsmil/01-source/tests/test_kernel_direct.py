#!/usr/bin/env python3
"""
Direct kernel module IOCTL testing for DSMIL driver
Bypasses 9.3 second SMI delays by using kernel direct access
"""

import ctypes
import fcntl
import struct
import time
from datetime import datetime
from pathlib import Path

# IOCTL command definitions from kernel module
MILDEV_IOC_MAGIC = ord('M')
MILDEV_IOC_GET_VERSION = 0x80044D01  # _IOR(MILDEV_IOC_MAGIC, 1, __u32)
MILDEV_IOC_GET_STATUS = 0x80204D02   # _IOR(MILDEV_IOC_MAGIC, 2, struct mildev_system_status)
MILDEV_IOC_SCAN_DEVICES = 0x80D84D03 # _IOR(MILDEV_IOC_MAGIC, 3, struct mildev_discovery_result)
MILDEV_IOC_READ_DEVICE = 0xC0104D04  # _IOWR(MILDEV_IOC_MAGIC, 4, struct mildev_device_info)
MILDEV_IOC_GET_THERMAL = 0x80044D05  # _IOR(MILDEV_IOC_MAGIC, 5, int)

# Structure definitions matching kernel module
class MildevSystemStatus(ctypes.Structure):
    _fields_ = [
        ('kernel_module_loaded', ctypes.c_uint8),
        ('thermal_safe', ctypes.c_uint8),
        ('current_temp_celsius', ctypes.c_int32),
        ('safe_device_count', ctypes.c_uint32),
        ('quarantined_count', ctypes.c_uint32),
        ('last_scan_timestamp', ctypes.c_uint64),
    ]

class MildevDeviceInfo(ctypes.Structure):
    _fields_ = [
        ('device_id', ctypes.c_uint16),
        ('state', ctypes.c_uint32),
        ('access', ctypes.c_uint32),
        ('is_quarantined', ctypes.c_uint8),
        ('last_response', ctypes.c_uint32),
        ('timestamp', ctypes.c_uint64),
        ('thermal_celsius', ctypes.c_int32),
    ]

class KernelDirectTester:
    def __init__(self):
        self.device_path = '/dev/dsmil-72dev'
        self.fd = None
        
        # Token mapping from test_tokens_with_module.py
        self.test_tokens = {
            'power': 0x0480,      # Position 0, Group 0 - SMI required
            'thermal': 0x0481,    # Position 1, Group 0 - Should be accessible
            'security': 0x0482,   # Position 2, Group 0 - Should be accessible  
            'memory': 0x0483,     # Position 3, Group 0 - SMI required
            'io': 0x0484,         # Position 4, Group 0 - Should be accessible
            'storage': 0x0486,    # Position 6, Group 0 - SMI required
            'sensor': 0x0489,     # Position 9, Group 0 - SMI required
        }
        
    def open_device(self):
        """Open the DSMIL kernel device"""
        try:
            self.fd = open(self.device_path, 'rb+', buffering=0)
            print(f"✅ Opened device: {self.device_path}")
            return True
        except PermissionError:
            print(f"❌ Permission denied: {self.device_path}")
            print("   Try: sudo python3 test_kernel_direct.py")
            return False
        except Exception as e:
            print(f"❌ Failed to open device: {e}")
            return False
    
    def close_device(self):
        """Close the device"""
        if self.fd:
            self.fd.close()
            self.fd = None
    
    def get_version(self):
        """Get kernel module version"""
        if not self.fd:
            return None
        
        try:
            version = ctypes.c_uint32()
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_VERSION, version)
            return version.value
        except Exception as e:
            print(f"❌ Get version failed: {e}")
            return None
    
    def get_system_status(self):
        """Get system status"""
        if not self.fd:
            return None
        
        try:
            status = MildevSystemStatus()
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_STATUS, status)
            return {
                'module_loaded': bool(status.kernel_module_loaded),
                'thermal_safe': bool(status.thermal_safe),
                'temperature': status.current_temp_celsius,
                'safe_devices': status.safe_device_count,
                'quarantined': status.quarantined_count,
                'last_scan': status.last_scan_timestamp
            }
        except Exception as e:
            print(f"❌ Get status failed: {e}")
            return None
    
    def get_thermal(self):
        """Get thermal temperature directly"""
        if not self.fd:
            return None
        
        try:
            temp = ctypes.c_int32()
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_THERMAL, temp)
            return temp.value
        except Exception as e:
            print(f"❌ Get thermal failed: {e}")
            return None
    
    def read_device(self, device_id):
        """Read specific device information"""
        if not self.fd:
            return None
        
        try:
            dev_info = MildevDeviceInfo()
            dev_info.device_id = device_id
            
            # Measure timing
            start_time = time.time()
            fcntl.ioctl(self.fd, MILDEV_IOC_READ_DEVICE, dev_info)
            end_time = time.time()
            
            return {
                'device_id': dev_info.device_id,
                'state': dev_info.state,
                'access': dev_info.access,
                'quarantined': bool(dev_info.is_quarantined),
                'last_response': dev_info.last_response,
                'timestamp': dev_info.timestamp,
                'thermal': dev_info.thermal_celsius,
                'response_time_ms': (end_time - start_time) * 1000
            }
        except Exception as e:
            print(f"❌ Read device 0x{device_id:04X} failed: {e}")
            return None
    
    def test_token_direct(self, name, token_id):
        """Test token access via kernel direct IOCTL"""
        print(f"\n🔍 Testing {name} token 0x{token_id:04X} via kernel direct")
        
        # Map SMBIOS token to device ID (simplified mapping)
        device_id = token_id & 0xFFFF
        
        result = self.read_device(device_id)
        if result:
            response_time = result['response_time_ms']
            if response_time < 100:  # Less than 100ms = direct access
                print(f"   ✅ FAST ACCESS: {response_time:.2f}ms")
                print(f"   📊 State: {result['state']}, Access: {result['access']}")
                print(f"   🔐 Quarantined: {result['quarantined']}")
                return {'accessible': True, 'fast': True, 'time_ms': response_time, 'data': result}
            else:
                print(f"   ⚠️  SLOW ACCESS: {response_time:.2f}ms (likely SMI)")
                return {'accessible': True, 'fast': False, 'time_ms': response_time, 'data': result}
        else:
            print(f"   ❌ NOT ACCESSIBLE")
            return {'accessible': False}
    
    def benchmark_performance(self):
        """Benchmark kernel direct vs SMI performance"""
        print("\n📊 PERFORMANCE BENCHMARK")
        print("="*50)
        
        # Test accessible tokens with kernel direct
        accessible_tokens = []
        smi_required_tokens = []
        
        for name, token_id in self.test_tokens.items():
            result = self.test_token_direct(name, token_id)
            if result.get('accessible'):
                if result.get('fast'):
                    accessible_tokens.append((name, result['time_ms']))
                else:
                    smi_required_tokens.append((name, result['time_ms']))
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Fast kernel access: {len(accessible_tokens)} tokens")
        print(f"   SMI required: {len(smi_required_tokens)} tokens")
        
        if accessible_tokens:
            avg_fast = sum(time for _, time in accessible_tokens) / len(accessible_tokens)
            print(f"   Average fast access: {avg_fast:.2f}ms")
        
        if smi_required_tokens:
            avg_slow = sum(time for _, time in smi_required_tokens) / len(smi_required_tokens)
            print(f"   Average SMI access: {avg_slow:.2f}ms")
            print(f"   Performance improvement: {avg_slow/avg_fast:.1f}x faster")
    
    def run_diagnostic(self):
        """Run comprehensive diagnostic"""
        print("="*60)
        print("KERNEL DIRECT ACCESS DIAGNOSTIC")
        print("="*60)
        
        if not self.open_device():
            return False
        
        try:
            # Test basic kernel functions
            print(f"🔧 Testing kernel module interface...")
            
            version = self.get_version()
            if version:
                print(f"   ✅ Version: 0x{version:08X}")
            else:
                print(f"   ❌ Version check failed")
                return False
            
            status = self.get_system_status()
            if status:
                print(f"   ✅ Status: Module={status['module_loaded']}, Thermal={status['thermal_safe']}")
                print(f"   📊 Temperature: {status['temperature']}°C")
                print(f"   📊 Safe devices: {status['safe_devices']}, Quarantined: {status['quarantined']}")
            else:
                print(f"   ❌ Status check failed")
            
            thermal = self.get_thermal()
            if thermal is not None:
                print(f"   ✅ Direct thermal: {thermal}°C")
            
            # Test token access
            self.benchmark_performance()
            
            return True
            
        finally:
            self.close_device()
    
    def debug_ioctl_calls(self):
        """Debug IOCTL call mechanics"""
        print("\n🔧 IOCTL DEBUGGING")
        print("="*40)
        
        if not self.open_device():
            return
        
        try:
            # Show IOCTL command values
            print(f"IOCTL Commands:")
            print(f"  GET_VERSION:  0x{MILDEV_IOC_GET_VERSION:08X}")
            print(f"  GET_STATUS:   0x{MILDEV_IOC_GET_STATUS:08X}")
            print(f"  READ_DEVICE:  0x{MILDEV_IOC_READ_DEVICE:08X}")
            print(f"  GET_THERMAL:  0x{MILDEV_IOC_GET_THERMAL:08X}")
            
            # Test each IOCTL
            print(f"\nTesting each IOCTL:")
            
            # Version test
            try:
                version = self.get_version()
                print(f"  ✅ GET_VERSION: 0x{version:08X}")
            except Exception as e:
                print(f"  ❌ GET_VERSION: {e}")
            
            # Status test  
            try:
                status = self.get_system_status()
                print(f"  ✅ GET_STATUS: {len(str(status))} bytes")
            except Exception as e:
                print(f"  ❌ GET_STATUS: {e}")
            
            # Thermal test
            try:
                thermal = self.get_thermal()
                print(f"  ✅ GET_THERMAL: {thermal}°C")
            except Exception as e:
                print(f"  ❌ GET_THERMAL: {e}")
            
            # Device read test
            try:
                device_info = self.read_device(0x0481)  # thermal token
                print(f"  ✅ READ_DEVICE: {device_info['response_time_ms']:.2f}ms")
            except Exception as e:
                print(f"  ❌ READ_DEVICE: {e}")
        
        finally:
            self.close_device()

def main():
    print("DSMIL Kernel Direct Access Tester")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    tester = KernelDirectTester()
    
    # Check if device exists
    if not Path(tester.device_path).exists():
        print(f"❌ Device not found: {tester.device_path}")
        print("   Load module with: sudo insmod dsmil-72dev.ko")
        return
    
    # Run diagnostics
    if tester.run_diagnostic():
        print("\n✅ Kernel direct access working!")
        print("   Use this method to avoid 9.3 second SMI delays")
    else:
        print("\n❌ Kernel direct access issues detected")
    
    # Debug IOCTL mechanics
    tester.debug_ioctl_calls()

if __name__ == "__main__":
    main()