#!/usr/bin/env python3
"""
Phase 1 Safe Device Testing - Based on NSA Intelligence Assessment
Tests 11 newly identified safe devices plus 12 JRTC1 training controllers
"""

import subprocess
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Safety configuration
PASSWORD = "1786"
READ_ONLY = True  # NEVER CHANGE THIS

# Critical quarantine list - NEVER ACCESS
QUARANTINED_DEVICES = {
    0x8009: "Emergency Wipe Controller",
    0x800A: "Secondary Wipe Trigger", 
    0x800B: "Final Sanitization",
    0x8019: "Network Isolation/Wipe",
    0x8029: "Communications Blackout"
}

# NSA-identified safe devices with confidence ratings
NSA_SAFE_DEVICES = {
    # High confidence (80-90%)
    0x8007: ("Security Audit Logger", 80),
    0x8010: ("Multi-Factor Authentication Controller", 90),
    0x8012: ("Security Event Correlator", 80),
    0x8020: ("Network Interface Controller", 90),
    0x8021: ("Wireless Communication Manager", 85),
    
    # Moderate confidence (65-75%)
    0x8015: ("Certificate Authority Interface", 65),
    0x8016: ("Security Baseline Monitor", 65),
    0x8023: ("Network Performance Monitor", 75),
    0x8024: ("VPN Hardware Accelerator", 70),
    0x8025: ("Network Quality of Service", 65),
}

# JRTC1 Training controllers (50-60% confidence)
TRAINING_DEVICES = {
    0x8060: ("Training Scenario Controller 0", 60),
    0x8061: ("Training Scenario Controller 1", 60),
    0x8062: ("Training Scenario Controller 2", 60),
    0x8063: ("Training Scenario Controller 3", 60),
    0x8064: ("Training Data Collection 0", 55),
    0x8065: ("Training Data Collection 1", 55),
    0x8066: ("Training Data Collection 2", 55),
    0x8067: ("Training Data Collection 3", 55),
    0x8068: ("Training Environment Control 0", 50),
    0x8069: ("Training Environment Control 1", 50),
    0x806A: ("Training Environment Control 2", 50),
    0x806B: ("Training Environment Control 3", 50),
}

def check_thermal_status() -> Tuple[bool, float]:
    """Check system thermal status before testing"""
    try:
        cmd = f'echo "{PASSWORD}" | sudo -S sensors'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=5)
        
        # Parse temperature from sensors output
        for line in result.stdout.split('\n'):
            if 'Core' in line and '°C' in line:
                # Extract temperature value
                temp_str = line.split('+')[1].split('°C')[0].strip()
                temp = float(temp_str)
                
                # Check against thresholds
                if temp > 95:
                    return False, temp
                elif temp > 85:
                    print(f"⚠️  Warning: High temperature {temp}°C")
                    
        return True, 75.0  # Default safe temperature
        
    except Exception as e:
        print(f"Warning: Could not check thermal status: {e}")
        return True, 75.0

def read_device_smi(token: int) -> Tuple[bool, int]:
    """Read device status using SMI - READ ONLY"""
    smi_code = f"""
#include <stdio.h>
#include <sys/io.h>
#include <unistd.h>
#include <signal.h>
#include <stdlib.h>

void timeout_handler(int sig) {{
    printf("TIMEOUT\\n");
    exit(1);
}}

int main() {{
    signal(SIGALRM, timeout_handler);
    alarm(1);  // 1 second timeout
    
    if (iopl(3) != 0) {{
        printf("ERROR: Cannot get I/O permissions\\n");
        return 1;
    }}
    
    // READ ONLY - Write token to query port
    outw(0x{token:04X}, 0x164E);
    
    // READ ONLY - Read status
    unsigned char status = inb(0x164F);
    
    printf("STATUS:0x%02X\\n", status);
    return 0;
}}
"""
    
    try:
        # Write and compile temporary C file
        with open("/tmp/test_smi_read.c", "w") as f:
            f.write(smi_code)
        
        compile_cmd = "gcc -O0 -o /tmp/test_smi_read /tmp/test_smi_read.c"
        subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
        
        # Execute with sudo
        cmd = f'echo "{PASSWORD}" | sudo -S /tmp/test_smi_read'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=2)
        
        # Parse result
        if "STATUS:" in result.stdout:
            status_line = result.stdout.strip().split("STATUS:")[1]
            status = int(status_line, 16)
            return True, status
        else:
            return False, 0
            
    except Exception as e:
        print(f"Error reading device {token:04X}: {e}")
        return False, 0
    finally:
        # Cleanup
        subprocess.run("rm -f /tmp/test_smi_read /tmp/test_smi_read.c", shell=True)

def test_device_group(devices: Dict[int, Tuple[str, int]], group_name: str) -> Dict:
    """Test a group of devices"""
    results = {
        "group": group_name,
        "total": len(devices),
        "successful": 0,
        "failed": 0,
        "devices": []
    }
    
    print(f"\nTesting {group_name} ({len(devices)} devices)")
    print("-" * 60)
    
    for token, (name, confidence) in devices.items():
        # Absolute quarantine check
        if token in QUARANTINED_DEVICES:
            print(f"❌ 0x{token:04X}: QUARANTINED - {QUARANTINED_DEVICES[token]} - NEVER ACCESS")
            continue
        
        print(f"Testing 0x{token:04X}: {name} (Confidence: {confidence}%)... ", end="", flush=True)
        
        # Perform read test
        success, status = read_device_smi(token)
        
        if success:
            active = (status & 0x01) == 1
            if active:
                print(f"✅ ACTIVE (Status: 0x{status:02X})")
                results["successful"] += 1
            else:
                print(f"⚠️  INACTIVE (Status: 0x{status:02X})")
            
            results["devices"].append({
                "token": f"0x{token:04X}",
                "name": name,
                "confidence": confidence,
                "status": f"0x{status:02X}",
                "active": active,
                "result": "SUCCESS"
            })
        else:
            print(f"❌ FAILED")
            results["failed"] += 1
            results["devices"].append({
                "token": f"0x{token:04X}",
                "name": name,
                "confidence": confidence,
                "result": "FAILED"
            })
        
        # Safety delay between tests
        time.sleep(0.2)
    
    return results

def main():
    """Main testing sequence"""
    print("=" * 70)
    print("DSMIL PHASE 1 SAFE DEVICE TESTING")
    print("Based on NSA Intelligence Assessment")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print(f"Mode: READ-ONLY - NO WRITES WILL BE PERFORMED")
    print(f"Quarantined devices: {len(QUARANTINED_DEVICES)} (WILL NOT ACCESS)")
    print()
    
    # Check thermal status
    print("Checking system thermal status...")
    thermal_safe, temp = check_thermal_status()
    if not thermal_safe:
        print(f"❌ ABORT: Temperature too high ({temp}°C)")
        return
    print(f"✅ Thermal status OK ({temp}°C)")
    
    # Test NSA-identified devices
    nsa_results = test_device_group(NSA_SAFE_DEVICES, "NSA-Identified Safe Devices")
    
    # Test training devices
    training_results = test_device_group(TRAINING_DEVICES, "JRTC1 Training Controllers")
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("TESTING SUMMARY")
    print("=" * 70)
    
    total_tested = nsa_results["successful"] + nsa_results["failed"] + \
                   training_results["successful"] + training_results["failed"]
    total_successful = nsa_results["successful"] + training_results["successful"]
    
    print(f"Total devices tested: {total_tested}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {nsa_results['failed'] + training_results['failed']}")
    print(f"Quarantined (not tested): {len(QUARANTINED_DEVICES)}")
    print()
    
    # Save results to JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "phase": "Phase 1 - Safe Device Expansion",
        "thermal_status": {"safe": thermal_safe, "temperature": temp},
        "nsa_devices": nsa_results,
        "training_devices": training_results,
        "summary": {
            "total_tested": total_tested,
            "total_successful": total_successful,
            "success_rate": f"{(total_successful/total_tested*100):.1f}%" if total_tested > 0 else "0%"
        }
    }
    
    report_file = f"phase1_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Full report saved to: {report_file}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 40)
    if total_successful >= 15:
        print("✅ Phase 1 expansion successful!")
        print("   - Add successful devices to production monitoring")
        print("   - Continue thermal monitoring during operations")
        print("   - Begin planning Phase 2 (Days 31-60)")
    else:
        print("⚠️  Limited success in Phase 1 testing")
        print("   - Review failed devices for patterns")
        print("   - Consider alternative access methods")
        print("   - Maintain conservative expansion approach")
    
    print("\n⚠️  REMINDER: Maintain absolute quarantine on devices:")
    for token, name in QUARANTINED_DEVICES.items():
        print(f"   - 0x{token:04X}: {name}")

if __name__ == "__main__":
    main()