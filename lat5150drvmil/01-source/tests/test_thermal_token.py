#!/usr/bin/env python3
"""
Test SMBIOS Token 0x481 - Thermal Control
Safe, monitored test with comprehensive logging
"""

import subprocess
import time
import json
from datetime import datetime
import sys
import psutil
import os

def get_thermal_zones():
    """Get all thermal zone temperatures"""
    temps = []
    for zone in os.listdir('/sys/class/thermal'):
        if zone.startswith('thermal_zone'):
            try:
                temp_file = f'/sys/class/thermal/{zone}/temp'
                with open(temp_file, 'r') as f:
                    temp = int(f.read().strip()) / 1000
                    temps.append(temp)
            except:
                pass
    return temps

def get_system_metrics():
    """Get comprehensive system metrics"""
    temps = get_thermal_zones()
    return {
        'timestamp': datetime.now().isoformat(),
        'thermal': {
            'max_temp': max(temps) if temps else 0,
            'avg_temp': sum(temps)/len(temps) if temps else 0,
            'all_temps': temps
        },
        'cpu': {
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 0
        },
        'memory': {
            'usage_percent': psutil.virtual_memory().percent,
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
    }

def read_token(token):
    """Read current token value"""
    try:
        cmd = f"sudo smbios-token-ctl --token-id={token:#x} --get 2>/dev/null"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Parse output for value
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'is' in line and ('Active' in line or 'Inactive' in line):
                    return 'Active' if 'Active' in line else 'Inactive'
        return None
    except Exception as e:
        return None

def set_token(token, value):
    """Set token value (true/false)"""
    try:
        action = '--activate' if value else '--deactivate'
        cmd = f"sudo smbios-token-ctl --token-id={token:#x} {action}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def monitor_kernel_logs():
    """Check for DSMIL activity in kernel logs"""
    try:
        cmd = "sudo dmesg | tail -50 | grep -i dsmil"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.stdout:
            return result.stdout.strip().split('\n')
    except:
        pass
    return []

def main():
    TOKEN = 0x481  # Thermal control token (90% confidence)
    MONITOR_TIME = 10  # seconds
    
    print("=" * 60)
    print("THERMAL CONTROL TOKEN TEST (0x481)")
    print("=" * 60)
    print(f"Target: Token {TOKEN:#x} - DSMIL Group 0, Device 1")
    print(f"Function: Thermal Control (90% confidence)")
    print(f"Test Duration: {MONITOR_TIME} seconds")
    print()
    
    # Initial metrics
    print("📊 Collecting baseline metrics...")
    baseline = get_system_metrics()
    initial_value = read_token(TOKEN)
    
    print(f"  Initial token state: {initial_value}")
    print(f"  Baseline temp: {baseline['thermal']['max_temp']:.1f}°C")
    print(f"  CPU usage: {baseline['cpu']['usage_percent']:.1f}%")
    print()
    
    # Safety check
    if baseline['thermal']['max_temp'] > 95:
        print("❌ Temperature too high for testing!")
        return 1
    
    results = {
        'token': TOKEN,
        'test_time': datetime.now().isoformat(),
        'baseline': baseline,
        'measurements': [],
        'kernel_activity': []
    }
    
    # Activate token
    print(f"🔧 Activating token {TOKEN:#x}...")
    if set_token(TOKEN, True):
        print("  ✅ Token activated")
        
        # Monitor for changes
        print(f"\n📈 Monitoring for {MONITOR_TIME} seconds...")
        for i in range(MONITOR_TIME):
            time.sleep(1)
            metrics = get_system_metrics()
            kernel_logs = monitor_kernel_logs()
            
            results['measurements'].append(metrics)
            if kernel_logs and kernel_logs not in results['kernel_activity']:
                results['kernel_activity'].extend(kernel_logs)
            
            # Display progress
            temp_change = metrics['thermal']['max_temp'] - baseline['thermal']['max_temp']
            print(f"  [{i+1:2d}/{MONITOR_TIME}] Temp: {metrics['thermal']['max_temp']:.1f}°C "
                  f"(Δ{temp_change:+.1f}°C) CPU: {metrics['cpu']['usage_percent']:.1f}%")
            
            # Safety check
            if metrics['thermal']['max_temp'] > 100:
                print("\n⚠️ Temperature limit reached! Deactivating...")
                break
        
        # Deactivate token
        print(f"\n🔧 Deactivating token {TOKEN:#x}...")
        if set_token(TOKEN, False):
            print("  ✅ Token deactivated")
        else:
            print("  ⚠️ Failed to deactivate token")
    else:
        print("  ❌ Failed to activate token")
        return 1
    
    # Final metrics
    time.sleep(2)
    final = get_system_metrics()
    results['final'] = final
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    if results['measurements']:
        max_temp = max(m['thermal']['max_temp'] for m in results['measurements'])
        avg_temp = sum(m['thermal']['max_temp'] for m in results['measurements']) / len(results['measurements'])
        temp_change = max_temp - baseline['thermal']['max_temp']
        
        print(f"Temperature Impact:")
        print(f"  Baseline: {baseline['thermal']['max_temp']:.1f}°C")
        print(f"  Maximum: {max_temp:.1f}°C")
        print(f"  Average: {avg_temp:.1f}°C")
        print(f"  Change: {temp_change:+.1f}°C")
        
        # Determine if token had effect
        if abs(temp_change) > 2.0:
            print(f"\n✅ TOKEN APPEARS TO CONTROL THERMAL MANAGEMENT")
            print(f"   Observed {temp_change:+.1f}°C change during activation")
        else:
            print(f"\n❓ TOKEN EFFECT UNCLEAR")
            print(f"   Temperature change within noise margin")
    
    if results['kernel_activity']:
        print(f"\n🔍 Kernel Activity Detected:")
        for log in results['kernel_activity'][:5]:
            print(f"  {log}")
    else:
        print(f"\n❌ No DSMIL kernel activity detected")
    
    # Save results
    output_file = f"token_{TOKEN:04x}_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📁 Detailed results saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    if os.geteuid() != 0:
        print("⚠️ This script requires root privileges for token control")
        print("Please run with: sudo python3 test_thermal_token.py")
        sys.exit(1)
    
    sys.exit(main())