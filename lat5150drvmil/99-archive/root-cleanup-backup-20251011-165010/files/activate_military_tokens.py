#!/usr/bin/env python3
"""
Military Token Activation Script for Dell Latitude 5450 MIL-SPEC
HARDWARE-DELL & SECURITY Agent Collaboration

MISSION: Safely activate discovered military tokens with comprehensive safety protocols
- Filters quarantined devices from activation list  
- Uses smbios-token-ctl for safe BIOS token control
- Activates Dell WMI security features
- Monitors thermal impact during operations
- Creates system checkpoints for rollback capability
- Verifies each activation before proceeding

SAFETY CRITICAL: Never activate quarantined devices
"""

import os
import sys
import json
import time
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class MilitaryTokenActivator:
    """Military-grade token activation with Dell-specific optimizations and security protocols"""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(f"checkpoints/activation_{self.timestamp}")
        self.log_file = Path(f"logs/activation_log_{self.timestamp}.txt")
        self.rollback_file = Path(f"rollback_data_{self.timestamp}.json")
        
        # Quarantined devices - NEVER ACTIVATE
        self.quarantined_tokens = {
            0x8009, 0x800A, 0x800B, 0x8019, 0x8029  # High-risk/unstable devices
        }
        
        # Target military tokens (filtered for safety)
        self.target_tokens = {
            0x8000: "Primary Command Interface",
            0x8014: "Secure Communications",
            0x801E: "Tactical Display Control", 
            0x8028: "Power Management Unit",
            0x8032: "Memory Protection",
            0x803C: "I/O Security Controller",
            0x8046: "Network Security Module",
            0x8050: "Storage Encryption",
            0x805A: "Sensor Array",
            0x8064: "Auxiliary Systems"
        }
        
        # Dell WMI security attributes for military systems
        self.wmi_security_features = [
            "SecureAdministrativeWorkstation",
            "TpmSecurity", 
            "SecureBoot",
            "ChasIntrusion",
            "FirmwareTamperDet",
            "ThermalManagement",
            "PowerWarn"
        ]
        
        # Safety thresholds
        self.max_temp_threshold = 95.0  # Critical thermal limit
        self.temp_warning_threshold = 90.0  # Warning threshold
        self.max_activation_attempts = 3
        self.thermal_cooldown_time = 30  # seconds
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'safety_checks': {},
            'activations': [],
            'wmi_activations': [],
            'thermal_monitoring': [],
            'failures': [],
            'rollback_data': {}
        }
        
        # Ensure directories exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(exist_ok=True)
    
    def _log(self, message: str, level: str = "INFO"):
        """Thread-safe logging with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
    
    def _get_system_info(self) -> Dict:
        """Collect system information for checkpoint"""
        try:
            info = {
                'hostname': subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip(),
                'kernel': subprocess.run(['uname', '-r'], capture_output=True, text=True).stdout.strip(),
                'product': "Dell Latitude 5450 MIL-SPEC",
                'dsmil_module_loaded': self._check_dsmil_module(),
                'initial_temperature': self._get_max_temperature()
            }
            return info
        except Exception as e:
            self._log(f"Error collecting system info: {e}", "ERROR")
            return {'error': str(e)}
    
    def _check_dsmil_module(self) -> bool:
        """Verify DSMIL kernel module is loaded"""
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            return 'dsmil_72dev' in result.stdout
        except:
            return False
    
    def _get_max_temperature(self) -> float:
        """Get current maximum system temperature"""
        max_temp = 0.0
        try:
            for zone_path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
                temp_raw = int(zone_path.read_text().strip())
                temp_celsius = temp_raw / 1000.0
                max_temp = max(max_temp, temp_celsius)
        except Exception as e:
            self._log(f"Error reading temperature: {e}", "WARN")
        return max_temp
    
    def _create_system_checkpoint(self) -> bool:
        """Create comprehensive system checkpoint for rollback"""
        self._log("Creating system checkpoint...")
        
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'system_state': {},
                'bios_tokens': {},
                'wmi_states': {},
                'thermal_baseline': self._get_max_temperature()
            }
            
            # Backup current SMBIOS token states
            for token_id in self.target_tokens.keys():
                try:
                    result = subprocess.run([
                        'sudo', '-S', 'smbios-token-ctl', 
                        f'--token-id=0x{token_id:04x}', '--get'
                    ], input="1786\n", text=True, capture_output=True, timeout=10)
                    
                    if result.returncode == 0:
                        checkpoint_data['bios_tokens'][f'0x{token_id:04x}'] = result.stdout.strip()
                except subprocess.TimeoutExpired:
                    self._log(f"Timeout reading token 0x{token_id:04x}", "WARN")
                except Exception as e:
                    self._log(f"Error reading token 0x{token_id:04x}: {e}", "WARN")
            
            # Backup WMI states
            for feature in self.wmi_security_features:
                wmi_path = f"/sys/devices/virtual/firmware-attributes/dell-wmi-sysman/attributes/{feature}"
                if Path(wmi_path).exists():
                    try:
                        current_value = Path(f"{wmi_path}/current_value").read_text().strip()
                        checkpoint_data['wmi_states'][feature] = current_value
                    except Exception as e:
                        self._log(f"Error reading WMI {feature}: {e}", "WARN")
            
            # Save checkpoint
            checkpoint_file = self.checkpoint_dir / "system_checkpoint.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Save rollback data
            with open(self.rollback_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.results['rollback_data'] = checkpoint_data
            self._log(f"Checkpoint created: {checkpoint_file}")
            return True
            
        except Exception as e:
            self._log(f"Failed to create checkpoint: {e}", "ERROR")
            return False
    
    def _check_thermal_safety(self) -> Tuple[bool, float]:
        """Check thermal conditions before activation"""
        current_temp = self._get_max_temperature()
        
        if current_temp >= self.max_temp_threshold:
            self._log(f"CRITICAL: Temperature {current_temp:.1f}°C exceeds maximum {self.max_temp_threshold}°C", "ERROR")
            return False, current_temp
        elif current_temp >= self.temp_warning_threshold:
            self._log(f"WARNING: Temperature {current_temp:.1f}°C approaching limit", "WARN")
            return True, current_temp
        else:
            self._log(f"Thermal OK: {current_temp:.1f}°C", "INFO")
            return True, current_temp
    
    def _verify_token_safety(self, token_id: int) -> bool:
        """Verify token is not in quarantined list"""
        if token_id in self.quarantined_tokens:
            self._log(f"BLOCKED: Token 0x{token_id:04x} is QUARANTINED", "ERROR")
            return False
        return True
    
    def _activate_smbios_token(self, token_id: int, description: str) -> bool:
        """Safely activate a single SMBIOS token"""
        self._log(f"Activating token 0x{token_id:04x}: {description}")
        
        # Safety checks
        if not self._verify_token_safety(token_id):
            return False
        
        thermal_ok, temp_before = self._check_thermal_safety()
        if not thermal_ok:
            self._log(f"Activation blocked due to thermal conditions", "ERROR")
            return False
        
        activation_result = {
            'token_id': f'0x{token_id:04x}',
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'temp_before': temp_before,
            'attempts': 0,
            'success': False
        }
        
        # Attempt activation with retries
        for attempt in range(1, self.max_activation_attempts + 1):
            activation_result['attempts'] = attempt
            
            try:
                # First, read current state
                read_result = subprocess.run([
                    'sudo', '-S', 'smbios-token-ctl', 
                    f'--token-id=0x{token_id:04x}', '--get'
                ], input="1786\n", text=True, capture_output=True, timeout=15)
                
                if read_result.returncode == 0:
                    current_state = read_result.stdout.strip()
                    activation_result['current_state'] = current_state
                    
                    if 'Active' in current_state:
                        self._log(f"Token 0x{token_id:04x} already active", "INFO")
                        activation_result['success'] = True
                        activation_result['action'] = 'already_active'
                        break
                    
                    # Attempt activation
                    activate_result = subprocess.run([
                        'sudo', '-S', 'smbios-token-ctl', 
                        f'--token-id=0x{token_id:04x}', '--activate'
                    ], input="1786\n", text=True, capture_output=True, timeout=15)
                    
                    if activate_result.returncode == 0:
                        # Verify activation
                        verify_result = subprocess.run([
                            'sudo', '-S', 'smbios-token-ctl', 
                            f'--token-id=0x{token_id:04x}', '--get'
                        ], input="1786\n", text=True, capture_output=True, timeout=10)
                        
                        if verify_result.returncode == 0 and 'Active' in verify_result.stdout:
                            self._log(f"✅ Successfully activated token 0x{token_id:04x}", "INFO")
                            activation_result['success'] = True
                            activation_result['action'] = 'activated'
                            activation_result['new_state'] = verify_result.stdout.strip()
                            break
                        else:
                            self._log(f"Activation verification failed for 0x{token_id:04x}", "WARN")
                    else:
                        self._log(f"Activation command failed: {activate_result.stderr}", "WARN")
                else:
                    self._log(f"Cannot read token 0x{token_id:04x}: {read_result.stderr}", "WARN")
                
                if attempt < self.max_activation_attempts:
                    self._log(f"Retrying activation (attempt {attempt + 1}/{self.max_activation_attempts})...")
                    time.sleep(2)
                    
            except subprocess.TimeoutExpired:
                self._log(f"Timeout during activation attempt {attempt}", "WARN")
            except Exception as e:
                self._log(f"Error during activation attempt {attempt}: {e}", "WARN")
        
        # Post-activation thermal check
        temp_after = self._get_max_temperature()
        activation_result['temp_after'] = temp_after
        activation_result['temp_change'] = temp_after - temp_before
        
        if activation_result['temp_change'] > 5.0:
            self._log(f"⚠️  Significant thermal increase: +{activation_result['temp_change']:.1f}°C", "WARN")
        
        self.results['activations'].append(activation_result)
        
        if not activation_result['success']:
            self.results['failures'].append(activation_result)
            self._log(f"❌ Failed to activate token 0x{token_id:04x}", "ERROR")
        
        return activation_result['success']
    
    def _activate_wmi_security(self) -> bool:
        """Activate Dell WMI security features"""
        self._log("Activating Dell WMI security features...")
        
        wmi_success_count = 0
        
        for feature in self.wmi_security_features:
            wmi_path = f"/sys/devices/virtual/firmware-attributes/dell-wmi-sysman/attributes/{feature}"
            
            wmi_result = {
                'feature': feature,
                'timestamp': datetime.now().isoformat(),
                'success': False
            }
            
            try:
                if Path(wmi_path).exists():
                    current_path = Path(f"{wmi_path}/current_value")
                    possible_path = Path(f"{wmi_path}/possible_values")
                    
                    if current_path.exists() and possible_path.exists():
                        current_value = current_path.read_text().strip()
                        possible_values = possible_path.read_text().strip().split('\n')
                        
                        wmi_result['current_value'] = current_value
                        wmi_result['possible_values'] = possible_values
                        
                        # Enable if not already enabled
                        if current_value.lower() in ['disabled', 'off', '0']:
                            if 'Enabled' in possible_values or 'On' in possible_values:
                                try:
                                    # Attempt to enable
                                    new_value_path = Path(f"{wmi_path}/new_value")
                                    enable_value = 'Enabled' if 'Enabled' in possible_values else 'On'
                                    new_value_path.write_text(enable_value)
                                    
                                    # Verify change
                                    time.sleep(1)
                                    updated_value = current_path.read_text().strip()
                                    
                                    if updated_value != current_value:
                                        self._log(f"✅ Enabled {feature}: {current_value} → {updated_value}", "INFO")
                                        wmi_result['success'] = True
                                        wmi_result['new_value'] = updated_value
                                        wmi_success_count += 1
                                    else:
                                        self._log(f"⚠️  {feature}: No change detected", "WARN")
                                        
                                except Exception as e:
                                    self._log(f"Error enabling {feature}: {e}", "WARN")
                        else:
                            self._log(f"✅ {feature} already enabled: {current_value}", "INFO")
                            wmi_result['success'] = True
                            wmi_success_count += 1
                    else:
                        self._log(f"WMI {feature}: Missing value files", "WARN")
                else:
                    self._log(f"WMI {feature}: Path not found", "WARN")
                    
            except Exception as e:
                self._log(f"Error with WMI feature {feature}: {e}", "WARN")
            
            self.results['wmi_activations'].append(wmi_result)
        
        success_rate = wmi_success_count / len(self.wmi_security_features) * 100
        self._log(f"WMI Security activation: {wmi_success_count}/{len(self.wmi_security_features)} ({success_rate:.1f}%) successful")
        
        return wmi_success_count > 0
    
    def _monitor_system_stability(self, duration_seconds: int = 60) -> bool:
        """Monitor system stability after activations"""
        self._log(f"Monitoring system stability for {duration_seconds} seconds...")
        
        start_time = time.time()
        monitoring_data = []
        
        while (time.time() - start_time) < duration_seconds:
            current_temp = self._get_max_temperature()
            timestamp = datetime.now().isoformat()
            
            monitoring_point = {
                'timestamp': timestamp,
                'temperature': current_temp,
                'elapsed': time.time() - start_time
            }
            
            monitoring_data.append(monitoring_point)
            self.results['thermal_monitoring'].append(monitoring_point)
            
            # Check for thermal issues
            if current_temp >= self.max_temp_threshold:
                self._log(f"CRITICAL: Thermal emergency at {current_temp:.1f}°C!", "ERROR")
                return False
            elif current_temp >= self.temp_warning_threshold:
                self._log(f"WARNING: High temperature {current_temp:.1f}°C", "WARN")
            
            time.sleep(10)  # Monitor every 10 seconds
        
        # Calculate stability metrics
        temps = [point['temperature'] for point in monitoring_data]
        avg_temp = sum(temps) / len(temps)
        max_temp = max(temps)
        temp_variance = max(temps) - min(temps)
        
        self._log(f"Stability monitoring complete:")
        self._log(f"  Average temperature: {avg_temp:.1f}°C")
        self._log(f"  Maximum temperature: {max_temp:.1f}°C")
        self._log(f"  Temperature variance: {temp_variance:.1f}°C")
        
        return temp_variance < 10.0 and max_temp < self.max_temp_threshold
    
    def execute_activation_sequence(self) -> bool:
        """Execute complete military token activation sequence"""
        self._log("="*80)
        self._log("MILITARY TOKEN ACTIVATION SEQUENCE INITIATED")
        self._log("Dell Latitude 5450 MIL-SPEC - HARDWARE-DELL & SECURITY Agents")
        self._log("="*80)
        
        # Phase 1: Safety checks and checkpoint creation
        self._log("Phase 1: Pre-activation safety checks...")
        
        if not self._check_dsmil_module():
            self._log("ERROR: DSMIL kernel module not loaded", "ERROR")
            self._log("Load with: sudo insmod 01-source/kernel/dsmil-72dev.ko", "ERROR")
            return False
        
        thermal_ok, initial_temp = self._check_thermal_safety()
        if not thermal_ok:
            self._log("Activation aborted due to thermal conditions", "ERROR")
            return False
        
        if not self._create_system_checkpoint():
            self._log("Activation aborted: Cannot create checkpoint", "ERROR")
            return False
        
        self._log(f"Initial system temperature: {initial_temp:.1f}°C")
        self._log("✅ Pre-activation checks passed")
        
        # Phase 2: SMBIOS token activation
        self._log("\nPhase 2: SMBIOS military token activation...")
        
        successful_activations = 0
        failed_activations = 0
        
        for token_id, description in self.target_tokens.items():
            # Thermal check before each activation
            thermal_ok, current_temp = self._check_thermal_safety()
            if not thermal_ok:
                self._log(f"Stopping activations due to thermal conditions: {current_temp:.1f}°C", "ERROR")
                break
            
            if self._activate_smbios_token(token_id, description):
                successful_activations += 1
            else:
                failed_activations += 1
            
            # Brief pause between activations
            time.sleep(3)
        
        # Phase 3: Dell WMI security activation
        self._log("\nPhase 3: Dell WMI security feature activation...")
        wmi_success = self._activate_wmi_security()
        
        # Phase 4: System stability monitoring
        self._log("\nPhase 4: System stability monitoring...")
        stability_ok = self._monitor_system_stability(60)
        
        # Phase 5: Final verification and reporting
        self._log("\nPhase 5: Final system verification...")
        
        final_temp = self._get_max_temperature()
        total_temp_change = final_temp - initial_temp
        
        # Generate comprehensive report
        self._log("="*80)
        self._log("ACTIVATION SEQUENCE COMPLETE")
        self._log("="*80)
        self._log(f"SMBIOS Token Results:")
        self._log(f"  Successful: {successful_activations}/{len(self.target_tokens)}")
        self._log(f"  Failed: {failed_activations}/{len(self.target_tokens)}")
        
        success_rate = successful_activations / len(self.target_tokens) * 100
        self._log(f"  Success Rate: {success_rate:.1f}%")
        
        self._log(f"\nThermal Analysis:")
        self._log(f"  Initial: {initial_temp:.1f}°C")
        self._log(f"  Final: {final_temp:.1f}°C")
        self._log(f"  Change: {total_temp_change:+.1f}°C")
        
        self._log(f"\nWMI Security: {'✅ Activated' if wmi_success else '❌ Failed'}")
        self._log(f"System Stability: {'✅ Stable' if stability_ok else '❌ Unstable'}")
        
        # Determine overall success
        overall_success = (
            successful_activations > 0 and 
            failed_activations == 0 and 
            stability_ok and 
            final_temp < self.max_temp_threshold
        )
        
        # Save results
        results_file = Path(f"logs/activation_results_{self.timestamp}.json")
        self.results['summary'] = {
            'overall_success': overall_success,
            'smbios_success_rate': success_rate,
            'wmi_activated': wmi_success,
            'system_stable': stability_ok,
            'total_temp_change': total_temp_change,
            'devices_expanded': successful_activations > (len(self.target_tokens) // 2)
        }
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self._log(f"📊 Complete results saved: {results_file}")
        
        if overall_success:
            self._log("🎯 MISSION ACCOMPLISHED: Military token activation successful", "SUCCESS")
            self._log(f"🔧 Device expansion achieved: {successful_activations} tokens activated")
        else:
            self._log("⚠️  MISSION PARTIAL: Some issues encountered", "WARN")
            self._log(f"💾 Rollback data available: {self.rollback_file}")
        
        return overall_success

def main():
    """Main execution function"""
    if os.geteuid() != 0:
        print("This script requires root privileges for SMBIOS token control")
        print("Run with: sudo python3 activate_military_tokens.py")
        sys.exit(1)
    
    # Verify system
    if not Path("/sys/devices/virtual/firmware-attributes/dell-wmi-sysman").exists():
        print("ERROR: Dell WMI interface not found")
        print("This script is designed for Dell military systems")
        sys.exit(1)
    
    try:
        activator = MilitaryTokenActivator()
        success = activator.execute_activation_sequence()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n🛑 Activation sequence interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Fatal error during activation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()