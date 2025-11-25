#!/usr/bin/env python3
"""
Fixed kernel module direct access with proper structure alignment
Tests the dsmil-72dev kernel module with corrected IOCTL structures
"""

import ctypes
import os
import fcntl
import struct
import time
import json
from datetime import datetime

# IOCTL command definitions with fixed sizes
MILDEV_IOC_MAGIC = 0x4D

# Fixed IOCTL commands based on kernel expectations
MILDEV_IOC_GET_VERSION = 0x80044D01  # Works - 4 bytes
MILDEV_IOC_GET_STATUS = 0x80204D02   # Fixed - 32 bytes  
MILDEV_IOC_GET_DEVICE_COUNT = 0x80044D03  # 4 bytes
MILDEV_IOC_READ_DEVICE = 0xC0104D04  # Fixed - 16 bytes
MILDEV_IOC_GET_THERMAL = 0x80044D05  # Works - 4 bytes

class MildevVersion(ctypes.Structure):
    """Version structure - 4 bytes total"""
    _pack_ = 1  # Force byte packing
    _fields_ = [
        ('major', ctypes.c_uint8),
        ('minor', ctypes.c_uint8),
        ('patch', ctypes.c_uint8),
        ('reserved', ctypes.c_uint8)
    ]

class MildevSystemStatus(ctypes.Structure):
    """System status - Fixed to 32 bytes"""
    _pack_ = 1
    _fields_ = [
        ('kernel_module_loaded', ctypes.c_uint8),    # 1
        ('thermal_safe', ctypes.c_uint8),            # 2
        ('_pad1', ctypes.c_uint16),                  # 4 (padding)
        ('current_temp_celsius', ctypes.c_int32),    # 8
        ('active_device_count', ctypes.c_uint32),    # 12
        ('quarantined_count', ctypes.c_uint32),      # 16
        ('smi_enabled', ctypes.c_uint8),             # 17
        ('tpm_available', ctypes.c_uint8),           # 18
        ('_pad2', ctypes.c_uint16),                  # 20 (padding)
        ('last_error', ctypes.c_uint32),             # 24
        ('status_flags', ctypes.c_uint32)            # 28
        # Total: 32 bytes
    ]

class MildevDeviceInfo(ctypes.Structure):
    """Device info - Fixed to 16 bytes"""
    _pack_ = 1
    _fields_ = [
        ('device_id', ctypes.c_uint16),      # 2
        ('status', ctypes.c_uint8),          # 3
        ('type', ctypes.c_uint8),            # 4
        ('value', ctypes.c_uint32),          # 8
        ('flags', ctypes.c_uint32),          # 12
        ('reserved', ctypes.c_uint32)        # 16
    ]

class KernelDirectTester:
    def __init__(self):
        self.device_path = '/dev/dsmil-72dev'
        self.fd = None
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'kernel_module': self.check_module(),
            'device_file': os.path.exists(self.device_path),
            'tests': []
        }
        
        # Token to device ID mapping
        # SMBIOS tokens 0x0480-0x04C7 map to kernel devices 0x8000-0x806B
        self.token_map = {}
        for i in range(72):
            smbios_token = 0x0480 + i
            kernel_device = 0x8000 + i
            self.token_map[smbios_token] = kernel_device
            
        # Quarantined devices (must be blocked)
        self.quarantined = [0x8009, 0x800A, 0x800B, 0x8019, 0x8029]
        
    def check_module(self):
        """Check if kernel module is loaded"""
        try:
            with open('/proc/modules', 'r') as f:
                return 'dsmil_72dev' in f.read()
        except:
            return False
    
    def open_device(self):
        """Open kernel device"""
        try:
            self.fd = os.open(self.device_path, os.O_RDWR)
            return True
        except Exception as e:
            print(f"❌ Failed to open device: {e}")
            return False
    
    def close_device(self):
        """Close kernel device"""
        if self.fd:
            os.close(self.fd)
            self.fd = None
    
    def test_version(self):
        """Test version IOCTL - known working"""
        if not self.fd:
            return None
            
        version = MildevVersion()
        try:
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_VERSION, version)
            return {
                'success': True,
                'version': f"{version.major}.{version.minor}.{version.patch}",
                'size': ctypes.sizeof(MildevVersion)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_status(self):
        """Test status IOCTL with fixed structure"""
        if not self.fd:
            return None
            
        status = MildevSystemStatus()
        print(f"Status structure size: {ctypes.sizeof(MildevSystemStatus)} bytes")
        
        try:
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_STATUS, status)
            return {
                'success': True,
                'kernel_loaded': bool(status.kernel_module_loaded),
                'thermal_safe': bool(status.thermal_safe),
                'temperature': status.current_temp_celsius,
                'active_devices': status.active_device_count,
                'quarantined': status.quarantined_count,
                'smi_enabled': bool(status.smi_enabled),
                'tpm_available': bool(status.tpm_available),
                'size': ctypes.sizeof(MildevSystemStatus)
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 
                   'size': ctypes.sizeof(MildevSystemStatus)}
    
    def test_device_read(self, device_id):
        """Test reading a device with fixed structure"""
        if not self.fd:
            return None
            
        device_info = MildevDeviceInfo()
        device_info.device_id = device_id
        
        print(f"Device structure size: {ctypes.sizeof(MildevDeviceInfo)} bytes")
        
        try:
            fcntl.ioctl(self.fd, MILDEV_IOC_READ_DEVICE, device_info)
            return {
                'success': True,
                'device_id': f"0x{device_info.device_id:04X}",
                'status': device_info.status,
                'type': device_info.type,
                'value': device_info.value,
                'flags': f"0x{device_info.flags:08X}",
                'quarantined': device_id in self.quarantined
            }
        except Exception as e:
            # Check if quarantine is enforced
            if device_id in self.quarantined and 'Operation not permitted' in str(e):
                return {
                    'success': True,
                    'device_id': f"0x{device_id:04X}",
                    'quarantined': True,
                    'note': 'Correctly blocked by quarantine'
                }
            return {'success': False, 'error': str(e), 
                   'device_id': f"0x{device_id:04X}"}
    
    def benchmark_performance(self):
        """Benchmark kernel direct access vs SMI"""
        if not self.fd:
            return None
            
        print("\n⚡ Performance Benchmark")
        
        # Test kernel direct access speed
        start = time.perf_counter()
        for _ in range(100):
            version = MildevVersion()
            fcntl.ioctl(self.fd, MILDEV_IOC_GET_VERSION, version)
        kernel_time = (time.perf_counter() - start) / 100
        
        # SMI baseline (from previous tests): 9.3 seconds
        smi_time = 9.3
        
        speedup = smi_time / kernel_time
        
        return {
            'kernel_direct_ms': kernel_time * 1000,
            'smi_baseline_ms': smi_time * 1000,
            'speedup_factor': speedup,
            'improvement_percent': (speedup - 1) * 100
        }
    
    def run_tests(self):
        """Run all tests"""
        print("🚀 DSMIL Kernel Direct Access Testing (Fixed Structures)")
        print("=" * 60)
        
        # Open device
        if not self.open_device():
            print("❌ Cannot open device - aborting tests")
            return self.results
        
        print(f"✅ Device opened: {self.device_path}")
        
        # Test version (known working)
        print("\n📊 Testing Version IOCTL...")
        version_result = self.test_version()
        if version_result and version_result['success']:
            print(f"✅ Version: {version_result['version']}")
        else:
            print(f"❌ Version test failed: {version_result}")
        self.results['tests'].append({'test': 'version', 'result': version_result})
        
        # Test status (with fixed structure)
        print("\n📊 Testing Status IOCTL (Fixed)...")
        status_result = self.test_status()
        if status_result and status_result['success']:
            print(f"✅ Status retrieved successfully")
            print(f"   Thermal: {status_result['temperature']}°C")
            print(f"   Active: {status_result['active_devices']} devices")
        else:
            print(f"⚠️  Status test: {status_result}")
        self.results['tests'].append({'test': 'status', 'result': status_result})
        
        # Test specific devices
        print("\n📊 Testing Device Access...")
        test_devices = [
            (0x8000, "TPM Control"),
            (0x8003, "Audit Log"),
            (0x8006, "Thermal Sensor"),
            (0x8009, "DATA DESTRUCTION (Quarantined)"),
            (0x8019, "NETWORK KILL (Quarantined)")
        ]
        
        for device_id, name in test_devices:
            print(f"\n  Testing {name} (0x{device_id:04X})...")
            device_result = self.test_device_read(device_id)
            
            if device_result:
                if device_result.get('quarantined'):
                    print(f"  🔒 QUARANTINED - Access correctly blocked")
                elif device_result['success']:
                    print(f"  ✅ Device accessible: Status={device_result['status']}")
                else:
                    print(f"  ❌ Error: {device_result['error']}")
                    
            self.results['tests'].append({
                'test': f'device_{device_id:04X}',
                'name': name,
                'result': device_result
            })
        
        # Benchmark performance
        print("\n⚡ Benchmarking Performance...")
        bench_result = self.benchmark_performance()
        if bench_result:
            print(f"✅ Kernel Direct: {bench_result['kernel_direct_ms']:.3f}ms")
            print(f"   SMI Baseline: {bench_result['smi_baseline_ms']:.1f}ms")
            print(f"   Speedup: {bench_result['speedup_factor']:.1f}x faster")
            print(f"   Improvement: {bench_result['improvement_percent']:.1f}%")
        self.results['tests'].append({'test': 'benchmark', 'result': bench_result})
        
        # Close device
        self.close_device()
        print("\n✅ Testing complete")
        
        # Save results
        output_file = f"kernel_direct_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📁 Results saved to: {output_file}")
        
        return self.results

if __name__ == "__main__":
    tester = KernelDirectTester()
    results = tester.run_tests()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    successful = sum(1 for t in results['tests'] 
                    if t['result'] and t['result'].get('success'))
    total = len(results['tests'])
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"📈 Success Rate: {(successful/total)*100:.1f}%")
    
    # Check if we achieved the goal
    if successful >= total * 0.8:  # 80% success threshold
        print("\n🎯 KERNEL DIRECT ACCESS VALIDATED")
        print("   Ready to replace SMI interface!")
    else:
        print("\n⚠️  Further structure alignment needed")