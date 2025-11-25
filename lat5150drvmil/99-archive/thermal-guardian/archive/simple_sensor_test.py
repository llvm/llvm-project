#!/usr/bin/env python3
"""
Simple sensor test for thermal guardian compatibility
"""

import os
import sys

# Thermal sensors to check
SENSORS = {
    'x86_pkg_temp': '/sys/class/thermal/thermal_zone9/temp',
    'dell_tcpu': '/sys/class/thermal/thermal_zone7/temp',
    'coretemp': '/sys/class/hwmon/hwmon7/temp1_input',
    'dell_cpu': '/sys/class/hwmon/hwmon5/temp1_input',
    'dell_smm': '/sys/class/hwmon/hwmon6/temp1_input'
}

# Control interfaces
CONTROLS = {
    'fan_control': '/sys/class/hwmon/hwmon6/pwm1',
    'cpu_freq': '/sys/devices/system/cpu/intel_pstate/max_perf_pct',
    'turbo': '/sys/devices/system/cpu/intel_pstate/no_turbo'
}

def test_sensors():
    print("🌡️ THERMAL SENSOR TEST")
    print("=" * 40)
    
    working_sensors = 0
    total_sensors = len(SENSORS)
    
    for name, path in SENSORS.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    temp_raw = f.read().strip()
                temp_celsius = float(temp_raw) / 1000.0
                
                status = "✓"
                if temp_celsius > 90:
                    status = "⚠️  HIGH"
                elif temp_celsius > 95:
                    status = "🚨 CRITICAL" 
                
                print(f"  {name:<15}: {temp_celsius:5.1f}°C {status}")
                working_sensors += 1
            else:
                print(f"  {name:<15}: NOT FOUND")
        except Exception as e:
            print(f"  {name:<15}: ERROR - {e}")
    
    print(f"\nSensors working: {working_sensors}/{total_sensors}")
    
    if working_sensors >= 3:
        print("✅ SENSOR TEST PASSED - Enough sensors for thermal guardian")
        return True
    else:
        print("❌ SENSOR TEST FAILED - Need at least 3 working sensors")
        return False

def test_controls():
    print("\n🎛️ CONTROL INTERFACE TEST")
    print("=" * 40)
    
    working_controls = 0
    total_controls = len(CONTROLS)
    
    for name, path in CONTROLS.items():
        try:
            if os.path.exists(path):
                readable = os.access(path, os.R_OK)
                writable = os.access(path, os.W_OK)
                
                status = ""
                if readable and writable:
                    status = "✅ READ/WRITE"
                    working_controls += 1
                elif readable:
                    status = "📖 READ ONLY (need root for write)"
                else:
                    status = "❌ NO ACCESS"
                
                # Try to read current value
                try:
                    with open(path, 'r') as f:
                        value = f.read().strip()
                    print(f"  {name:<12}: {status} (current: {value})")
                except:
                    print(f"  {name:<12}: {status}")
            else:
                print(f"  {name:<12}: NOT FOUND")
        except Exception as e:
            print(f"  {name:<12}: ERROR - {e}")
    
    print(f"\nControls accessible: {working_controls}/{total_controls}")
    
    if working_controls >= 1:
        print("⚠️  CONTROL TEST PARTIAL - Some controls need root access")
        return True
    else:
        print("❌ CONTROL TEST FAILED - No controls accessible")
        return False

def show_current_status():
    print("\n📊 CURRENT THERMAL STATUS")
    print("=" * 40)
    
    # Find highest temperature
    max_temp = 0
    max_sensor = ""
    
    for name, path in SENSORS.items():
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    temp_raw = f.read().strip()
                temp_celsius = float(temp_raw) / 1000.0
                
                if temp_celsius > max_temp:
                    max_temp = temp_celsius
                    max_sensor = name
        except:
            continue
    
    print(f"Highest temperature: {max_temp:.1f}°C ({max_sensor})")
    
    if max_temp > 95:
        print("🚨 SYSTEM CRITICAL - Thermal guardian needed IMMEDIATELY")
    elif max_temp > 85:
        print("⚠️  SYSTEM HOT - Thermal guardian recommended")
    elif max_temp > 75:
        print("🟡 SYSTEM WARM - Thermal guardian beneficial")
    else:
        print("✅ SYSTEM COOL - Thermal guardian ready for deployment")
    
    # Check fan status
    fan_path = CONTROLS['fan_control']
    if os.path.exists(fan_path):
        try:
            with open(fan_path, 'r') as f:
                pwm = int(f.read().strip())
            fan_percent = (pwm / 255) * 100
            print(f"Current fan speed: {fan_percent:.0f}% (PWM: {pwm})")
        except:
            print("Fan status: Unknown")

def main():
    print("THERMAL GUARDIAN COMPATIBILITY CHECK")
    print("====================================")
    print("Dell LAT5150DRVMIL Thermal System Test")
    print()
    
    sensors_ok = test_sensors()
    controls_ok = test_controls()
    show_current_status()
    
    print("\n" + "=" * 40)
    print("COMPATIBILITY SUMMARY")
    print("=" * 40)
    
    if sensors_ok and controls_ok:
        print("✅ COMPATIBLE - Ready for thermal guardian deployment")
        print("\nNext steps:")
        print("1. Run as root: sudo ./quick_thermal_test.sh")  
        print("2. Deploy: sudo ./deploy_thermal_guardian.sh")
        return 0
    elif sensors_ok:
        print("⚠️  PARTIAL COMPATIBILITY - Sensors OK, controls need root")
        print("\nNext steps:")
        print("1. Run as root for full test: sudo ./quick_thermal_test.sh")
        return 1
    else:
        print("❌ INCOMPATIBLE - Sensor issues detected")
        print("\nTroubleshooting:")
        print("1. Check if thermal drivers are loaded")
        print("2. Verify system compatibility")
        return 2

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)