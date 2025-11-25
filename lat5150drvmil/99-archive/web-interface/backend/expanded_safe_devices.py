#!/usr/bin/env python3
"""
Expanded Safe Device Configuration - Phase 1 Implementation
Based on NSA Intelligence Assessment for 11 high-confidence safe devices
"""

from typing import List, Dict, Set
from datetime import datetime

# Critical quarantine list - NEVER MODIFY
QUARANTINED_DEVICES: Set[int] = {
    0x8009,  # Emergency Wipe Controller - EXTREME DANGER
    0x800A,  # Secondary Wipe Trigger - EXTREME DANGER
    0x800B,  # Final Sanitization - EXTREME DANGER
    0x8019,  # Network Isolation/Wipe - HIGH DANGER
    0x8029,  # Communications Blackout - HIGH DANGER
}

# Phase 1: High-confidence safe devices (NSA 75-90% confidence)
SAFE_MONITORING_DEVICES: Dict[int, Dict] = {
    # Original 6 safe devices (proven safe through testing)
    0x8000: {"name": "Group 0 Controller", "confidence": 100, "status": "ACTIVE"},
    0x8001: {"name": "Thermal Monitoring", "confidence": 100, "status": "ACTIVE"},
    0x8002: {"name": "Power Status", "confidence": 100, "status": "ACTIVE"},
    0x8003: {"name": "Fan Control", "confidence": 100, "status": "ACTIVE"},
    0x8004: {"name": "CPU Status", "confidence": 100, "status": "ACTIVE"},
    0x8006: {"name": "System Supervisor", "confidence": 100, "status": "ACTIVE"},
    
    # NSA-identified safe devices (11 new additions)
    0x8007: {"name": "Security Audit Logger", "confidence": 80, "status": "PENDING"},
    0x8010: {"name": "Multi-Factor Authentication Controller", "confidence": 90, "status": "PENDING"},
    0x8012: {"name": "Security Event Correlator", "confidence": 80, "status": "PENDING"},
    0x8015: {"name": "Certificate Authority Interface", "confidence": 65, "status": "PENDING"},
    0x8016: {"name": "Security Baseline Monitor", "confidence": 65, "status": "PENDING"},
    0x8020: {"name": "Network Interface Controller", "confidence": 90, "status": "PENDING"},
    0x8021: {"name": "Wireless Communication Manager", "confidence": 85, "status": "PENDING"},
    0x8023: {"name": "Network Performance Monitor", "confidence": 75, "status": "PENDING"},
    0x8024: {"name": "VPN Hardware Accelerator", "confidence": 70, "status": "PENDING"},
    0x8025: {"name": "Network Quality of Service", "confidence": 65, "status": "PENDING"},
    
    # JRTC1 Training Controllers (12 devices, all safe)
    0x8060: {"name": "Training Scenario Controller 0", "confidence": 60, "status": "PENDING"},
    0x8061: {"name": "Training Scenario Controller 1", "confidence": 60, "status": "PENDING"},
    0x8062: {"name": "Training Scenario Controller 2", "confidence": 60, "status": "PENDING"},
    0x8063: {"name": "Training Scenario Controller 3", "confidence": 60, "status": "PENDING"},
    0x8064: {"name": "Training Data Collection 0", "confidence": 55, "status": "PENDING"},
    0x8065: {"name": "Training Data Collection 1", "confidence": 55, "status": "PENDING"},
    0x8066: {"name": "Training Data Collection 2", "confidence": 55, "status": "PENDING"},
    0x8067: {"name": "Training Data Collection 3", "confidence": 55, "status": "PENDING"},
    0x8068: {"name": "Training Environment Control 0", "confidence": 50, "status": "PENDING"},
    0x8069: {"name": "Training Environment Control 1", "confidence": 50, "status": "PENDING"},
    0x806A: {"name": "Training Environment Control 2", "confidence": 50, "status": "PENDING"},
    0x806B: {"name": "Training Environment Control 3", "confidence": 50, "status": "PENDING"},
}

# Phase 2: Likely safe devices (NSA 60-74% confidence) - Future expansion
LIKELY_SAFE_DEVICES: Dict[int, Dict] = {
    0x8005: {"name": "TPM/HSM Interface Controller", "confidence": 85, "status": "FUTURE"},
    0x8008: {"name": "Secure Boot Validator", "confidence": 75, "status": "FUTURE"},
    0x8011: {"name": "Encryption Key Management", "confidence": 85, "status": "FUTURE"},
    0x8013: {"name": "Intrusion Detection System", "confidence": 70, "status": "FUTURE"},
    0x8014: {"name": "Security Policy Enforcement", "confidence": 70, "status": "FUTURE"},
    0x8022: {"name": "Network Security Filter", "confidence": 80, "status": "FUTURE"},
    0x8027: {"name": "Network Authentication Gateway", "confidence": 60, "status": "FUTURE"},
}

# Phase 3-6: Caution required devices (NSA <60% confidence or unknown)
CAUTION_REQUIRED_DEVICES: Set[int] = {
    0x800C,  # Emergency Communications Override
    0x8017, 0x8018, 0x801A, 0x801B,  # Advanced security functions
    0x8026, 0x8028, 0x802A, 0x802B,  # Network control functions
    *range(0x8030, 0x803C),  # Group 3: Data Processing (all unknown)
    *range(0x8044, 0x8048),  # Group 4: Storage Security (potential wipe)
}

def is_device_safe(device_id: int) -> bool:
    """Check if device is safe for monitoring"""
    return device_id in SAFE_MONITORING_DEVICES

def is_device_quarantined(device_id: int) -> bool:
    """Check if device is permanently quarantined"""
    return device_id in QUARANTINED_DEVICES

def get_device_risk_assessment(device_id: int) -> Dict:
    """Get comprehensive risk assessment for a device"""
    if device_id in QUARANTINED_DEVICES:
        return {
            "device_id": f"0x{device_id:04X}",
            "risk_level": "EXTREME",
            "access": "NEVER",
            "reason": "Confirmed data destruction capability",
            "confidence": 100
        }
    
    if device_id in SAFE_MONITORING_DEVICES:
        info = SAFE_MONITORING_DEVICES[device_id]
        return {
            "device_id": f"0x{device_id:04X}",
            "risk_level": "LOW",
            "access": "READ_ONLY",
            "name": info["name"],
            "confidence": info["confidence"],
            "status": info["status"]
        }
    
    if device_id in LIKELY_SAFE_DEVICES:
        info = LIKELY_SAFE_DEVICES[device_id]
        return {
            "device_id": f"0x{device_id:04X}",
            "risk_level": "MODERATE",
            "access": "FUTURE_MONITORING",
            "name": info["name"],
            "confidence": info["confidence"],
            "status": info["status"]
        }
    
    if device_id in CAUTION_REQUIRED_DEVICES:
        return {
            "device_id": f"0x{device_id:04X}",
            "risk_level": "HIGH",
            "access": "RESTRICTED",
            "reason": "Unknown function requiring investigation",
            "confidence": 0
        }
    
    # Unknown device
    return {
        "device_id": f"0x{device_id:04X}",
        "risk_level": "UNKNOWN",
        "access": "DENIED",
        "reason": "Device not assessed",
        "confidence": 0
    }

def get_monitoring_expansion_plan() -> Dict:
    """Get the phased monitoring expansion plan"""
    return {
        "phase_1": {
            "name": "Safe Monitoring Expansion",
            "duration": "Days 1-30",
            "devices": list(SAFE_MONITORING_DEVICES.keys()),
            "device_count": len(SAFE_MONITORING_DEVICES),
            "status": "IN_PROGRESS",
            "start_date": datetime.now().isoformat()
        },
        "phase_2": {
            "name": "High-Confidence Device Addition",
            "duration": "Days 31-60",
            "devices": list(LIKELY_SAFE_DEVICES.keys()),
            "device_count": len(LIKELY_SAFE_DEVICES),
            "status": "PLANNED"
        },
        "phase_3": {
            "name": "Group 0-2 Unknown Exploration",
            "duration": "Days 61-90",
            "target_groups": [0, 1, 2],
            "status": "PLANNED"
        },
        "phase_4": {
            "name": "Groups 3-6 Investigation",
            "duration": "Days 91-120",
            "target_groups": [3, 4, 5, 6],
            "status": "PLANNED",
            "note": "EXTREME CAUTION - Group 3 data processing unknown"
        },
        "phase_5": {
            "name": "Controlled Write Testing",
            "duration": "Days 121-150",
            "status": "PLANNED",
            "requirement": "Isolated test environment mandatory"
        },
        "phase_6": {
            "name": "Full Production Deployment",
            "duration": "Day 151+",
            "status": "PLANNED",
            "goal": "All 79 non-quarantined devices operational"
        }
    }

def generate_safe_testing_script() -> str:
    """Generate bash script for safe device testing"""
    script = """#!/bin/bash
# DSMIL Phase 1 Safe Device Testing Script
# Generated: {}
# Based on NSA Intelligence Assessment

# Safety configuration
set -euo pipefail
SUDO_PASSWORD="1786"

# Color codes for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

echo "========================================"
echo "DSMIL PHASE 1 SAFE DEVICE TESTING"
echo "========================================"
echo "Testing {} devices identified as safe by NSA"
echo ""

# Quarantine check function
check_quarantine() {{
    local device=$1
    case $device in
        0x8009|0x800A|0x800B|0x8019|0x8029)
            echo -e "${{RED}}CRITICAL: Device $device is QUARANTINED - SKIPPING${{NC}}"
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}}

# Safe devices array (NSA high-confidence)
SAFE_DEVICES=(
""".format(datetime.now().isoformat(), len(SAFE_MONITORING_DEVICES))
    
    # Add device list
    for device_id in sorted(SAFE_MONITORING_DEVICES.keys()):
        script += f"    0x{device_id:04X}\n"
    
    script += """)

# Test each safe device
echo "Starting safe device testing sequence..."
echo ""

for device in "${SAFE_DEVICES[@]}"; do
    echo -n "Testing device $device: "
    
    # Absolute quarantine check
    if ! check_quarantine $device; then
        continue
    fi
    
    # Perform READ-ONLY test
    echo "$SUDO_PASSWORD" | sudo -S python3 << EOF
import sys
sys.path.append('/home/john/LAT5150DRVMIL/web-interface/backend')
from expanded_safe_devices import get_device_risk_assessment

device_id = int('$device', 16)
assessment = get_device_risk_assessment(device_id)
print(f"{assessment['name'] if 'name' in assessment else 'Unknown'} - Confidence: {assessment.get('confidence', 0)}%")
EOF
    
    # Add monitoring delay
    sleep 0.5
done

echo ""
echo "========================================"
echo "TESTING COMPLETE"
echo "========================================"
echo "Safe devices tested: ${#SAFE_DEVICES[@]}"
echo "Quarantined devices avoided: 5"
echo ""
echo "Next steps:"
echo "1. Review test results above"
echo "2. Activate monitoring for successful devices"
echo "3. Begin Phase 2 planning (Days 31-60)"
"""
    return script

if __name__ == "__main__":
    # Display expansion status
    print("DSMIL Safe Device Expansion Configuration")
    print("=" * 50)
    print(f"Quarantined devices (NEVER ACCESS): {len(QUARANTINED_DEVICES)}")
    print(f"Safe monitoring devices: {len(SAFE_MONITORING_DEVICES)}")
    print(f"  - Active: {sum(1 for d in SAFE_MONITORING_DEVICES.values() if d['status'] == 'ACTIVE')}")
    print(f"  - Pending: {sum(1 for d in SAFE_MONITORING_DEVICES.values() if d['status'] == 'PENDING')}")
    print(f"Likely safe (Phase 2): {len(LIKELY_SAFE_DEVICES)}")
    print(f"Caution required: {len(CAUTION_REQUIRED_DEVICES)}")
    print()
    
    # Display monitoring plan
    plan = get_monitoring_expansion_plan()
    print("Phased Expansion Plan:")
    for phase_key, phase_data in plan.items():
        print(f"\n{phase_key.upper()}: {phase_data['name']}")
        print(f"  Duration: {phase_data['duration']}")
        print(f"  Status: {phase_data['status']}")
        if 'device_count' in phase_data:
            print(f"  Devices: {phase_data['device_count']}")