#!/usr/bin/env python3
"""
DSMIL Token Mapping Orchestrator
Systematically maps all 72 tokens and records results
"""

import json
import time
import subprocess
import os
import sys
from datetime import datetime
from pathlib import Path

# Add database path
sys.path.append(str(Path(__file__).parent / 'database'))
sys.path.append(str(Path(__file__).parent / 'database' / 'scripts'))

try:
    from database.backends.database_backend import DatabaseBackend
except:
    print("Note: Database backend not available, using JSON logging")
    DatabaseBackend = None

class DSMILTokenMapper:
    def __init__(self):
        self.work_dir = Path(__file__).parent
        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize database if available
        self.db = DatabaseBackend() if DatabaseBackend else None
        
        # Token ranges for all 6 groups
        self.token_ranges = {
            'Group_0': list(range(0x0480, 0x048C)),  # 12 tokens
            'Group_1': list(range(0x048C, 0x0498)),  # 12 tokens
            'Group_2': list(range(0x0498, 0x04A4)),  # 12 tokens
            'Group_3': list(range(0x04A4, 0x04B0)),  # 12 tokens
            'Group_4': list(range(0x04B0, 0x04BC)),  # 12 tokens
            'Group_5': list(range(0x04BC, 0x04C8))   # 12 tokens
        }
        
        # Locked token positions (0,3,6,9)
        self.locked_positions = [0, 3, 6, 9]
        
        # Mapping results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_info(),
            'tokens': {},
            'groups': {},
            'patterns': {},
            'thermal_impact': {},
            'summary': {}
        }
        
    def get_system_info(self):
        """Get system information"""
        info = {
            'hostname': os.uname().nodename,
            'kernel': os.uname().release,
            'arch': os.uname().machine,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get Dell system info
        try:
            product = Path('/sys/devices/virtual/dmi/id/product_name').read_text().strip()
            bios = Path('/sys/devices/virtual/dmi/id/bios_version').read_text().strip()
            info['product'] = product
            info['bios'] = bios
        except:
            pass
            
        return info
    
    def get_thermal_reading(self):
        """Get current thermal readings"""
        temps = []
        for zone in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
            try:
                temp = int(zone.read_text().strip()) / 1000
                temps.append(temp)
            except:
                pass
        return max(temps) if temps else 0
    
    def check_module_loaded(self):
        """Check if DSMIL module is loaded"""
        try:
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            return 'dsmil_72dev' in result.stdout
        except:
            return False
    
    def read_token_smbios(self, token):
        """Read token value via SMBIOS (for accessible tokens)"""
        try:
            cmd = f"smbios-token-ctl --token-id={token:#x} --get 2>/dev/null"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # Parse output
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'is' in line:
                        if 'Active' in line:
                            return 'Active'
                        elif 'Inactive' in line:
                            return 'Inactive'
            return None
        except:
            return None
    
    def check_kernel_messages(self, pattern='dsmil'):
        """Check kernel messages for pattern"""
        try:
            cmd = f"dmesg | tail -50 | grep -i {pattern}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                return result.stdout.strip().split('\n')
        except:
            pass
        return []
    
    def map_single_token(self, token, group_name, position):
        """Map a single token"""
        print(f"\n📍 Mapping token 0x{token:04X} (Group {group_name}, Position {position})")
        
        # Record start conditions
        start_temp = self.get_thermal_reading()
        start_time = time.time()
        
        token_info = {
            'token_id': f'0x{token:04X}',
            'group': group_name,
            'position': position,
            'is_locked': position in self.locked_positions,
            'start_temp': start_temp,
            'timestamp': datetime.now().isoformat()
        }
        
        # Determine access method
        if position in self.locked_positions:
            print(f"   ⚠️  Locked token (position {position}) - requires SMI/kernel access")
            token_info['access_method'] = 'SMI'
            token_info['accessible'] = False
            
            # Check for kernel support
            kernel_msgs = self.check_kernel_messages(f'0x{token:04x}')
            if kernel_msgs:
                print(f"   📝 Kernel activity detected:")
                for msg in kernel_msgs[:3]:
                    print(f"      {msg}")
                token_info['kernel_activity'] = kernel_msgs
        else:
            # Try SMBIOS access
            print(f"   🔍 Attempting SMBIOS access...")
            value = self.read_token_smbios(token)
            
            if value is not None:
                print(f"   ✅ Accessible via SMBIOS: {value}")
                token_info['access_method'] = 'SMBIOS'
                token_info['current_value'] = value
                token_info['accessible'] = True
            else:
                print(f"   ❌ Not accessible via SMBIOS")
                token_info['access_method'] = 'None'
                token_info['accessible'] = False
        
        # Record end conditions
        end_temp = self.get_thermal_reading()
        elapsed = time.time() - start_time
        
        token_info['end_temp'] = end_temp
        token_info['temp_change'] = end_temp - start_temp
        token_info['elapsed_time'] = elapsed
        
        # Hypothesize function based on position
        functions = {
            0: "Power Management",
            1: "Thermal Control",
            2: "Security Module",
            3: "Memory Controller",
            4: "I/O Controller",
            5: "Network Interface",
            6: "Storage Controller",
            7: "Display Control",
            8: "Audio Control",
            9: "Sensor Hub",
            10: "Accelerometer",
            11: "Unknown/Reserved"
        }
        
        token_info['hypothesized_function'] = functions.get(position, "Unknown")
        
        # Store result
        self.results['tokens'][f'0x{token:04X}'] = token_info
        
        # Small delay for thermal recovery
        time.sleep(0.5)
        
        return token_info
    
    def map_group(self, group_name, tokens):
        """Map all tokens in a group"""
        print(f"\n{'='*60}")
        print(f"Mapping {group_name}: {len(tokens)} tokens")
        print(f"Token range: 0x{tokens[0]:04X} - 0x{tokens[-1]:04X}")
        print(f"{'='*60}")
        
        group_results = {
            'name': group_name,
            'token_count': len(tokens),
            'accessible_count': 0,
            'locked_count': 0,
            'tokens': [],
            'start_time': datetime.now().isoformat()
        }
        
        for i, token in enumerate(tokens):
            result = self.map_single_token(token, group_name, i)
            group_results['tokens'].append(result)
            
            if result['accessible']:
                group_results['accessible_count'] += 1
            if result['is_locked']:
                group_results['locked_count'] += 1
        
        group_results['end_time'] = datetime.now().isoformat()
        self.results['groups'][group_name] = group_results
        
        # Summary for group
        print(f"\n📊 {group_name} Summary:")
        print(f"   Accessible: {group_results['accessible_count']}/{len(tokens)}")
        print(f"   Locked: {group_results['locked_count']}/{len(tokens)}")
        
        return group_results
    
    def detect_patterns(self):
        """Detect patterns in token mapping"""
        patterns = {
            'locked_pattern': "Every 3rd token (positions 0,3,6,9)",
            'access_distribution': {},
            'function_mapping': {},
            'thermal_impact': {}
        }
        
        # Analyze access patterns
        total_accessible = 0
        total_locked = 0
        
        for token_id, info in self.results['tokens'].items():
            if info['accessible']:
                total_accessible += 1
            if info['is_locked']:
                total_locked += 1
            
            # Group by function
            func = info['hypothesized_function']
            if func not in patterns['function_mapping']:
                patterns['function_mapping'][func] = []
            patterns['function_mapping'][func].append(token_id)
        
        patterns['access_distribution'] = {
            'accessible': total_accessible,
            'locked': total_locked,
            'total': len(self.results['tokens']),
            'accessibility_rate': f"{(total_accessible/72)*100:.1f}%"
        }
        
        self.results['patterns'] = patterns
        return patterns
    
    def generate_report(self):
        """Generate comprehensive mapping report"""
        print("\n" + "="*70)
        print("DSMIL TOKEN MAPPING COMPLETE")
        print("="*70)
        
        # Overall statistics
        patterns = self.detect_patterns()
        
        print(f"\n📊 Overall Statistics:")
        print(f"   Total tokens mapped: {patterns['access_distribution']['total']}")
        print(f"   Accessible tokens: {patterns['access_distribution']['accessible']}")
        print(f"   Locked tokens: {patterns['access_distribution']['locked']}")
        print(f"   Accessibility rate: {patterns['access_distribution']['accessibility_rate']}")
        
        print(f"\n🔍 Discovered Pattern:")
        print(f"   {patterns['locked_pattern']}")
        
        print(f"\n📋 Function Distribution:")
        for func, tokens in patterns['function_mapping'].items():
            print(f"   {func}: {len(tokens)} tokens")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON report
        json_file = self.log_dir / f'token_mapping_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 JSON report saved: {json_file}")
        
        # CSV export
        csv_file = self.log_dir / f'token_mapping_{timestamp}.csv'
        with open(csv_file, 'w') as f:
            f.write("Token,Group,Position,Locked,Accessible,Function,Access Method\n")
            for token_id, info in self.results['tokens'].items():
                f.write(f"{token_id},{info['group']},{info['position']},"
                       f"{info['is_locked']},{info['accessible']},"
                       f"{info['hypothesized_function']},{info['access_method']}\n")
        print(f"📊 CSV export saved: {csv_file}")
        
        # Summary text
        summary_file = self.log_dir / f'mapping_summary_{timestamp}.txt'
        with open(summary_file, 'w') as f:
            f.write("DSMIL TOKEN MAPPING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {self.results['timestamp']}\n")
            f.write(f"System: {self.results['system'].get('product', 'Unknown')}\n")
            f.write(f"BIOS: {self.results['system'].get('bios', 'Unknown')}\n\n")
            
            f.write(f"Total Tokens: {patterns['access_distribution']['total']}\n")
            f.write(f"Accessible: {patterns['access_distribution']['accessible']}\n")
            f.write(f"Locked: {patterns['access_distribution']['locked']}\n")
            f.write(f"Rate: {patterns['access_distribution']['accessibility_rate']}\n\n")
            
            f.write("Locked Token Pattern:\n")
            f.write(f"  {patterns['locked_pattern']}\n\n")
            
            f.write("Function Mapping:\n")
            for func, tokens in patterns['function_mapping'].items():
                f.write(f"  {func}: {len(tokens)} tokens\n")
                f.write(f"    {', '.join(tokens[:5])}")
                if len(tokens) > 5:
                    f.write("...")
                f.write("\n")
        
        print(f"📝 Summary saved: {summary_file}")
        
        return self.results
    
    def run_mapping(self):
        """Run complete token mapping"""
        print("\n" + "="*70)
        print("DSMIL TOKEN MAPPING SYSTEM")
        print("="*70)
        print(f"System: {self.results['system'].get('product', 'Unknown')}")
        print(f"BIOS: {self.results['system'].get('bios', 'Unknown')}")
        print(f"Start time: {self.results['timestamp']}")
        
        # Check if module is loaded
        if not self.check_module_loaded():
            print("\n⚠️  DSMIL kernel module not loaded")
            print("   Note: Only SMBIOS-accessible tokens will be mapped")
            print("   To load module: sudo insmod dsmil-72dev.ko")
        else:
            print("\n✅ DSMIL kernel module detected")
        
        # Initial thermal reading
        initial_temp = self.get_thermal_reading()
        print(f"\n🌡️  Initial temperature: {initial_temp:.1f}°C")
        
        if initial_temp > 95:
            print("❌ Temperature too high for safe mapping!")
            return None
        
        # Map each group
        for group_name, tokens in self.token_ranges.items():
            self.map_group(group_name, tokens)
            
            # Check thermal between groups
            current_temp = self.get_thermal_reading()
            print(f"\n🌡️  Current temperature: {current_temp:.1f}°C")
            
            if current_temp > 95:
                print("⚠️  Temperature threshold reached, pausing...")
                time.sleep(30)
        
        # Generate final report
        return self.generate_report()

def main():
    """Main entry point"""
    mapper = DSMILTokenMapper()
    
    print("🚀 Starting DSMIL Token Mapping")
    print("   This will map all 72 tokens across 6 groups")
    print("   Locked tokens will be identified for kernel access")
    print("")
    
    # Run the mapping
    results = mapper.run_mapping()
    
    if results:
        print("\n✅ Mapping completed successfully!")
        print(f"   Total tokens processed: {len(results['tokens'])}")
    else:
        print("\n❌ Mapping aborted due to safety limits")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())