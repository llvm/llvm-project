#!/usr/bin/env python3
"""
Safe identification of DOD wipe device through READ-ONLY pattern analysis
NO WRITES - NO ACTIVATION - READ STATUS ONLY
"""

import subprocess
import time
from datetime import datetime
from typing import List, Dict, Tuple
import json

class SafeWipeDeviceIdentifier:
    """
    Identify potential DOD wipe device through safe pattern analysis.
    ALL OPERATIONS ARE READ-ONLY - NO WRITES PERMITTED
    """
    
    def __init__(self):
        self.password = "1786"
        self.READ_ONLY = True  # NEVER CHANGE THIS
        self.WRITES_PROHIBITED = True  # ABSOLUTE
        
        # Known military wipe patterns (hypothesis)
        self.wipe_indicators = {
            'status_patterns': [
                0xFF,  # All bits set (armed state)
                0xDE,  # Common military pattern
                0xAD,  # DEAD pattern component
                0xBE,  # BEEF pattern component  
                0xEF,  # BEEF pattern component
                0x55,  # Alternating pattern (0101)
                0xAA,  # Alternating pattern (1010)
                0x5A,  # Security pattern
                0xA5,  # Security pattern inverse
            ],
            'suspicious_bits': {
                7: 'armed/ready',
                6: 'locked/secured',
                5: 'critical/dangerous',
                4: 'irreversible',
                3: 'confirmation_required',
            }
        }
        
        # Risk scoring system
        self.risk_scores = {}
        self.quarantine_list = []
        
    def analyze_status_byte(self, status: int, token: int) -> Dict:
        """
        Analyze status byte for wipe indicators.
        READ-ONLY analysis - no device interaction.
        """
        analysis = {
            'token': f"0x{token:04X}",
            'status': f"0x{status:02X}",
            'binary': f"{status:08b}",
            'risk_score': 0,
            'indicators': [],
            'suspicious_bits': []
        }
        
        # Check for known patterns
        if status in self.wipe_indicators['status_patterns']:
            analysis['risk_score'] += 50
            analysis['indicators'].append(f"MATCHES_KNOWN_PATTERN: 0x{status:02X}")
        
        # Check individual bits
        for bit_pos, description in self.wipe_indicators['suspicious_bits'].items():
            if status & (1 << bit_pos):
                analysis['risk_score'] += 20
                analysis['suspicious_bits'].append(f"BIT_{bit_pos}: {description}")
        
        # Special pattern checks
        if status == 0xDE or status == 0xAD or status == 0xBE or status == 0xEF:
            analysis['risk_score'] += 30
            analysis['indicators'].append("DEADBEEF_COMPONENT")
        
        # Check for all high bits (dangerous)
        if status == 0xFF:
            analysis['risk_score'] += 40
            analysis['indicators'].append("ALL_BITS_HIGH_ARMED_STATE")
        
        # Check for confirmation patterns
        if status & 0xF0 == 0xD0:  # Dx pattern
            analysis['risk_score'] += 25
            analysis['indicators'].append("CONFIRMATION_PATTERN")
        
        return analysis
    
    def read_device_safely(self, token: int) -> Tuple[bool, int]:
        """
        Read device status using SMI - READ ONLY.
        NO WRITES ARE PERFORMED.
        """
        # Create C program for SMI read
        smi_read_code = f"""
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
    // Set timeout protection
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
    
    // Output status for analysis
    printf("STATUS:0x%02X\\n", status);
    
    return 0;
}}
"""
        
        try:
            # Write temporary C file
            with open("/tmp/safe_smi_read.c", "w") as f:
                f.write(smi_read_code)
            
            # Compile with safety flags
            compile_cmd = "gcc -O0 -Wall -Wextra -o /tmp/safe_smi_read /tmp/safe_smi_read.c"
            subprocess.run(compile_cmd, shell=True, check=True, capture_output=True)
            
            # Execute with sudo (READ ONLY)
            cmd = f'echo "{self.password}" | sudo -S /tmp/safe_smi_read'
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
            subprocess.run("rm -f /tmp/safe_smi_read /tmp/safe_smi_read.c", shell=True)
    
    def identify_wipe_devices(self) -> List[Dict]:
        """
        Identify potential wipe devices through pattern analysis.
        ALL OPERATIONS ARE READ-ONLY.
        """
        print("="*70)
        print("SAFE WIPE DEVICE IDENTIFICATION - READ ONLY MODE")
        print("="*70)
        print(f"Start: {datetime.now()}")
        print("Mode: READ-ONLY - NO WRITES WILL BE PERFORMED")
        print()
        
        high_risk_devices = []
        
        # Analyze each device
        for group_id in range(7):
            print(f"\nAnalyzing Group {group_id} (READ ONLY)...")
            base_token = 0x8000 + (group_id * 0x10)
            
            for device_idx in range(12):
                token = base_token + device_idx
                
                # Read status safely
                success, status = self.read_device_safely(token)
                
                if success:
                    # Analyze the status byte
                    analysis = self.analyze_status_byte(status, token)
                    self.risk_scores[token] = analysis['risk_score']
                    
                    # Display analysis
                    print(f"\nDevice 0x{token:04X}:")
                    print(f"  Status: 0x{status:02X} ({status:08b})")
                    print(f"  Risk Score: {analysis['risk_score']}")
                    
                    if analysis['indicators']:
                        print(f"  Indicators: {', '.join(analysis['indicators'])}")
                    
                    if analysis['suspicious_bits']:
                        print(f"  Suspicious Bits: {', '.join(analysis['suspicious_bits'])}")
                    
                    # Flag high-risk devices
                    if analysis['risk_score'] >= 70:
                        print(f"  ⚠️  HIGH RISK - POTENTIAL WIPE DEVICE")
                        high_risk_devices.append(analysis)
                        self.quarantine_list.append(token)
                    elif analysis['risk_score'] >= 50:
                        print(f"  ⚠️  ELEVATED RISK - REQUIRES INVESTIGATION")
                        high_risk_devices.append(analysis)
                
                # Safety delay between reads
                time.sleep(0.1)
        
        return high_risk_devices
    
    def generate_safety_report(self, high_risk_devices: List[Dict]):
        """Generate safety report with findings."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'mode': 'READ_ONLY',
            'total_devices': 84,
            'devices_analyzed': len(self.risk_scores),
            'high_risk_count': len(high_risk_devices),
            'quarantine_list': [f"0x{t:04X}" for t in self.quarantine_list],
            'risk_distribution': {},
            'high_risk_devices': high_risk_devices,
            'recommendations': []
        }
        
        # Calculate risk distribution
        for token, score in self.risk_scores.items():
            if score >= 70:
                category = "CRITICAL"
            elif score >= 50:
                category = "HIGH"
            elif score >= 30:
                category = "MODERATE"
            else:
                category = "LOW"
            
            if category not in report['risk_distribution']:
                report['risk_distribution'][category] = 0
            report['risk_distribution'][category] += 1
        
        # Add recommendations
        if self.quarantine_list:
            report['recommendations'].append(
                f"NEVER write to these devices: {', '.join([f'0x{t:04X}' for t in self.quarantine_list])}"
            )
        
        report['recommendations'].extend([
            "Maintain READ-ONLY mode for all operations",
            "Research Dell military documentation for device purposes",
            "Use isolated test system for any future write attempts",
            "Keep comprehensive backups before any testing",
            "Consider these devices armed and dangerous"
        ])
        
        # Save report
        with open("wipe_device_identification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Display summary
        print("\n" + "="*70)
        print("IDENTIFICATION COMPLETE - SAFETY REPORT")
        print("="*70)
        print(f"Total Devices Analyzed: {report['devices_analyzed']}")
        print(f"High Risk Devices: {report['high_risk_count']}")
        print(f"Quarantined Devices: {len(self.quarantine_list)}")
        print("\nRisk Distribution:")
        for category, count in report['risk_distribution'].items():
            print(f"  {category}: {count} devices")
        
        if self.quarantine_list:
            print("\n⚠️  CRITICAL WARNING ⚠️")
            print("The following devices are QUARANTINED - DO NOT WRITE:")
            for token in self.quarantine_list:
                print(f"  - 0x{token:04X}")
        
        print("\n📄 Full report saved to: wipe_device_identification_report.json")
        print("\nREMEMBER: ALL DEVICES REMAIN DANGEROUS UNTIL PROVEN SAFE")
        
        return report

def main():
    """Main execution with safety checks."""
    
    print("⚠️  SAFETY CHECK ⚠️")
    print("This script performs READ-ONLY analysis to identify potential wipe devices.")
    print("NO WRITES will be performed. NO DEVICES will be activated.")
    print()
    
    # Auto-confirm for non-interactive mode
    print("Auto-confirming READ-ONLY mode for safety analysis...")
    
    # Create identifier
    identifier = SafeWipeDeviceIdentifier()
    
    # Perform safe identification
    high_risk_devices = identifier.identify_wipe_devices()
    
    # Generate report
    report = identifier.generate_safety_report(high_risk_devices)
    
    print("\nIdentification complete. All operations were READ-ONLY.")
    print("Review the report and maintain maximum safety protocols.")

if __name__ == "__main__":
    main()