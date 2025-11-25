#!/usr/bin/env python3
"""
Debian Trixie Compatibility Layer for SMBIOS Token Testing
=========================================================

Handles distribution-specific differences between Ubuntu 24.04 and Debian Trixie
for the Dell Latitude 5450 MIL-SPEC SMBIOS token testing framework.

Key Differences:
- Package management (apt vs apt)
- SMBIOS library paths and versions
- Kernel module compilation paths
- System service management
- Thermal monitoring interfaces

Author: TESTBED Agent
Version: 1.0.0
Date: 2025-09-01
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DistroInfo:
    """Distribution information"""
    name: str
    version: str
    kernel_version: str
    smbios_package: str
    smbios_tools_package: str
    headers_package: str
    build_essential_package: str
    python_smbios_package: str

class DebianCompatibility:
    """Debian Trixie compatibility management"""
    
    def __init__(self):
        self.distro_info = self._detect_distribution()
        self.compatibility_data = self._load_compatibility_data()
        
    def _detect_distribution(self) -> DistroInfo:
        """Detect current distribution and version"""
        
        # Read /etc/os-release
        os_release = {}
        try:
            with open('/etc/os-release') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        os_release[key] = value.strip('"')
        except FileNotFoundError:
            pass
            
        # Get kernel version
        kernel_version = os.uname().release
        
        # Determine distribution
        if os_release.get('ID') == 'ubuntu':
            return DistroInfo(
                name='Ubuntu',
                version=os_release.get('VERSION_ID', '24.04'),
                kernel_version=kernel_version,
                smbios_package='libsmbios-dev',
                smbios_tools_package='libsmbios-bin',
                headers_package=f'linux-headers-{kernel_version}',
                build_essential_package='build-essential',
                python_smbios_package='python3-libsmbios'
            )
        elif os_release.get('ID') == 'debian':
            return DistroInfo(
                name='Debian',
                version=os_release.get('VERSION_ID', '13'),
                kernel_version=kernel_version,
                smbios_package='libsmbios-dev',
                smbios_tools_package='smbios-utils',
                headers_package=f'linux-headers-{kernel_version}',
                build_essential_package='build-essential',
                python_smbios_package='python3-libsmbios'
            )
        else:
            # Default to Ubuntu-like
            return DistroInfo(
                name='Unknown',
                version='unknown',
                kernel_version=kernel_version,
                smbios_package='libsmbios-dev',
                smbios_tools_package='libsmbios-bin',
                headers_package=f'linux-headers-{kernel_version}',
                build_essential_package='build-essential',
                python_smbios_package='python3-libsmbios'
            )
            
    def _load_compatibility_data(self) -> Dict:
        """Load distribution-specific compatibility data"""
        
        return {
            'Ubuntu': {
                'smbios_paths': [
                    '/sys/module/dell_smbios',
                    '/sys/devices/platform/dell-smbios'
                ],
                'thermal_paths': [
                    '/sys/class/thermal',
                    '/sys/devices/virtual/thermal'
                ],
                'kernel_module_path': '/lib/modules',
                'service_manager': 'systemd',
                'package_manager': 'apt',
                'smbios_token_cmd': 'smbios-token-ctl'
            },
            'Debian': {
                'smbios_paths': [
                    '/sys/devices/platform/dell-smbios',
                    '/sys/module/dell_smbios'  # Different order
                ],
                'thermal_paths': [
                    '/sys/class/thermal',
                    '/sys/devices/virtual/thermal'
                ],
                'kernel_module_path': '/lib/modules',
                'service_manager': 'systemd',
                'package_manager': 'apt',
                'smbios_token_cmd': 'smbios-token-ctl'  # Same in recent Debian
            }
        }
        
    def check_system_compatibility(self) -> Tuple[bool, List[str]]:
        """Check if system is compatible with token testing"""
        
        issues = []
        
        print(f"📋 Checking compatibility for {self.distro_info.name} {self.distro_info.version}")
        
        # Check kernel version compatibility
        kernel_parts = self.distro_info.kernel_version.split('.')
        try:
            major = int(kernel_parts[0])
            minor = int(kernel_parts[1])
            
            if major < 6 or (major == 6 and minor < 10):
                issues.append(f"Kernel version {self.distro_info.kernel_version} may be too old (need 6.10+)")
        except (ValueError, IndexError):
            issues.append(f"Could not parse kernel version: {self.distro_info.kernel_version}")
            
        # Check for SMBIOS tools
        smbios_cmd = self.compatibility_data.get(self.distro_info.name, {}).get('smbios_token_cmd', 'smbios-token-ctl')
        
        try:
            result = subprocess.run(['which', smbios_cmd], capture_output=True)
            if result.returncode != 0:
                issues.append(f"SMBIOS tool '{smbios_cmd}' not found - install {self.distro_info.smbios_tools_package}")
        except Exception as e:
            issues.append(f"Could not check for SMBIOS tools: {e}")
            
        # Check for build tools (needed for kernel module)
        try:
            result = subprocess.run(['which', 'gcc'], capture_output=True)
            if result.returncode != 0:
                issues.append(f"GCC not found - install {self.distro_info.build_essential_package}")
        except Exception as e:
            issues.append(f"Could not check for build tools: {e}")
            
        # Check for kernel headers
        headers_path = Path(f"/lib/modules/{self.distro_info.kernel_version}/build")
        if not headers_path.exists():
            issues.append(f"Kernel headers not found - install {self.distro_info.headers_package}")
            
        # Check for Dell SMBIOS module support
        dell_modules = ['dell_smbios', 'dell_wmi']
        for module in dell_modules:
            try:
                result = subprocess.run(['modinfo', module], capture_output=True)
                if result.returncode != 0:
                    issues.append(f"Dell module '{module}' not available")
            except Exception:
                issues.append(f"Could not check Dell module '{module}'")
                
        return len(issues) == 0, issues
        
    def install_dependencies(self) -> bool:
        """Install required dependencies for the current distribution"""
        
        print(f"📦 Installing dependencies for {self.distro_info.name}...")
        
        packages = [
            self.distro_info.build_essential_package,
            self.distro_info.smbios_package,
            self.distro_info.smbios_tools_package,
            self.distro_info.headers_package,
            'python3-pip',
            'python3-psutil'
        ]
        
        # Add distribution-specific packages
        if self.distro_info.name == 'Debian':
            packages.extend(['dkms', 'module-assistant'])
        elif self.distro_info.name == 'Ubuntu':
            packages.extend(['dkms'])
            
        try:
            # Update package list
            result = subprocess.run(['sudo', 'apt', 'update'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ Package list update failed: {result.stderr}")
                
            # Install packages
            cmd = ['sudo', 'apt', 'install', '-y'] + packages
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
                
                # Install Python packages
                python_packages = ['psutil']
                pip_cmd = [sys.executable, '-m', 'pip', 'install', '--user'] + python_packages
                subprocess.run(pip_cmd, capture_output=True)
                
                return True
            else:
                print(f"❌ Package installation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Dependency installation error: {e}")
            return False
            
    def get_smbios_command(self, action: str, token: str, value: str = None) -> List[str]:
        """Get the correct SMBIOS command for the current distribution"""
        
        base_cmd = ['sudo', 'smbios-token-ctl']
        
        if action == 'get':
            return base_cmd + ['--get-token', token]
        elif action == 'set':
            if value is None:
                raise ValueError("Value required for set action")
            return base_cmd + ['--set-token', token, '--value', value]
        else:
            raise ValueError(f"Unknown action: {action}")
            
    def get_thermal_sensors(self) -> List[Path]:
        """Get thermal sensor paths for the current distribution"""
        
        thermal_sensors = []
        
        # Standard thermal zone path (works on both)
        thermal_dir = Path("/sys/class/thermal")
        if thermal_dir.exists():
            thermal_sensors.extend(thermal_dir.glob("thermal_zone*"))
            
        # Distribution-specific paths
        distro_paths = self.compatibility_data.get(self.distro_info.name, {}).get('thermal_paths', [])
        for path_str in distro_paths:
            path = Path(path_str)
            if path.exists() and path not in [p.parent for p in thermal_sensors]:
                thermal_sensors.extend(path.glob("thermal_zone*"))
                
        return sorted(set(thermal_sensors))
        
    def create_distribution_makefile(self, output_path: Path):
        """Create distribution-specific Makefile for kernel module"""
        
        makefile_content = f"""# Distribution-specific Makefile for DSMIL kernel module
# Generated for {self.distro_info.name} {self.distro_info.version}

KERNEL_VERSION := {self.distro_info.kernel_version}
KDIR := /lib/modules/$(KERNEL_VERSION)/build
PWD := $(shell pwd)

# Distribution detection
DISTRO := {self.distro_info.name}

obj-m += dsmil-72dev.o

# Distribution-specific flags
ifeq ($(DISTRO),Debian)
    ccflags-y += -DDEBIAN_BUILD
    ccflags-y += -DTHERMAL_THRESHOLD_DEFAULT=100
endif

ifeq ($(DISTRO),Ubuntu)
    ccflags-y += -DUBUNTU_BUILD
    ccflags-y += -DTHERMAL_THRESHOLD_DEFAULT=100
endif

# Common flags
ccflags-y += -DFORCE_JRTC1_MODE
ccflags-y += -Wall -Wno-unused-function

all:
\tmake -C $(KDIR) M=$(PWD) modules

clean:
\tmake -C $(KDIR) M=$(PWD) clean

install:
\tmake -C $(KDIR) M=$(PWD) modules_install

# Distribution-specific install
install-{self.distro_info.name.lower()}:
\tsudo insmod dsmil-72dev.ko
\techo "DSMIL module loaded on {self.distro_info.name}"

uninstall:
\tsudo rmmod dsmil-72dev || true

# Compatibility check
check-compat:
\t@echo "Distribution: {self.distro_info.name} {self.distro_info.version}"
\t@echo "Kernel: {self.distro_info.kernel_version}"
\t@echo "Headers: $(KDIR)"
\t@test -d $(KDIR) || echo "ERROR: Kernel headers not found!"
\t@which gcc > /dev/null || echo "ERROR: GCC not found!"

.PHONY: all clean install uninstall check-compat install-{self.distro_info.name.lower()}
"""
        
        with open(output_path, 'w') as f:
            f.write(makefile_content)
            
        print(f"📝 Created {self.distro_info.name}-specific Makefile: {output_path}")
        
    def create_install_script(self, output_path: Path):
        """Create distribution-specific installation script"""
        
        script_content = f"""#!/bin/bash
# Distribution-specific installation script
# Generated for {self.distro_info.name} {self.distro_info.version}

set -e

DISTRO="{self.distro_info.name}"
VERSION="{self.distro_info.version}"
KERNEL="{self.distro_info.kernel_version}"

echo "🔧 Installing DSMIL testing framework on $DISTRO $VERSION"

# Check root privileges for system operations
if [[ $EUID -eq 0 ]]; then
   echo "⚠️ Running as root. Some operations will be performed with reduced privileges."
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y \\
    {self.distro_info.build_essential_package} \\
    {self.distro_info.smbios_package} \\
    {self.distro_info.smbios_tools_package} \\
    {self.distro_info.headers_package} \\
    python3-pip \\
    python3-psutil \\
    dkms

# Distribution-specific packages
if [[ "$DISTRO" == "Debian" ]]; then
    sudo apt install -y module-assistant
    echo "✅ Debian-specific packages installed"
fi

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
python3 -m pip install --user psutil

# Load Dell modules
echo "🖥️ Loading Dell SMBIOS modules..."
sudo modprobe dell-smbios || echo "⚠️ dell-smbios module not available"
sudo modprobe dell-wmi || echo "⚠️ dell-wmi module not available"

# Verify SMBIOS tools
echo "🔍 Verifying SMBIOS tools..."
if command -v smbios-token-ctl &> /dev/null; then
    echo "✅ smbios-token-ctl found"
    sudo smbios-token-ctl --version || true
else
    echo "❌ smbios-token-ctl not found!"
    exit 1
fi

# Check thermal sensors
echo "🌡️ Checking thermal sensors..."
THERMAL_COUNT=$(find /sys/class/thermal -name "thermal_zone*" 2>/dev/null | wc -l)
echo "Found $THERMAL_COUNT thermal sensors"

# Check kernel headers
echo "🔨 Verifying kernel build environment..."
if [[ -d "/lib/modules/$KERNEL/build" ]]; then
    echo "✅ Kernel headers found"
else
    echo "❌ Kernel headers not found at /lib/modules/$KERNEL/build"
    echo "   Install: sudo apt install {self.distro_info.headers_package}"
    exit 1
fi

# Test GCC
echo "⚙️ Testing compiler..."
if gcc --version &> /dev/null; then
    echo "✅ GCC compiler available"
else
    echo "❌ GCC compiler not found!"
    exit 1
fi

echo ""
echo "🎉 Installation complete for $DISTRO $VERSION!"
echo ""
echo "Next steps:"
echo "1. cd 01-source/kernel"
echo "2. make -f Makefile.{self.distro_info.name.lower()}"
echo "3. sudo insmod dsmil-72dev.ko"
echo "4. python3 ../../testing/smbios_testbed_framework.py"
echo ""
"""

        with open(output_path, 'w') as f:
            f.write(script_content)
            
        output_path.chmod(0o755)
        print(f"📝 Created {self.distro_info.name}-specific install script: {output_path}")
        
    def validate_smbios_access(self) -> Tuple[bool, List[str]]:
        """Validate SMBIOS access for the current distribution"""
        
        issues = []
        
        # Test SMBIOS command access
        try:
            cmd = self.get_smbios_command('get', '0x8013')  # Test with known military token
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                if "permission denied" in result.stderr.lower():
                    issues.append("SMBIOS access requires sudo privileges")
                elif "not found" in result.stderr.lower():
                    issues.append(f"Token 0x8013 not found (expected for non-MIL-SPEC system)")
                else:
                    issues.append(f"SMBIOS command failed: {result.stderr}")
            else:
                print("✅ SMBIOS access validated")
                
        except subprocess.TimeoutExpired:
            issues.append("SMBIOS command timeout - system may be unresponsive")
        except Exception as e:
            issues.append(f"SMBIOS validation error: {e}")
            
        # Check SMBIOS paths
        smbios_paths = self.compatibility_data.get(self.distro_info.name, {}).get('smbios_paths', [])
        found_paths = []
        
        for path_str in smbios_paths:
            path = Path(path_str)
            if path.exists():
                found_paths.append(str(path))
                
        if not found_paths:
            issues.append(f"No SMBIOS sysfs paths found in {smbios_paths}")
        else:
            print(f"✅ SMBIOS sysfs paths found: {found_paths}")
            
        return len(issues) == 0, issues

def main():
    """Test Debian compatibility system"""
    
    print("🧪 Debian Compatibility Test")
    print("=" * 40)
    
    compat = DebianCompatibility()
    
    print(f"📋 Detected: {compat.distro_info.name} {compat.distro_info.version}")
    print(f"🔧 Kernel: {compat.distro_info.kernel_version}")
    print()
    
    # Check system compatibility
    print("🔍 Checking system compatibility...")
    is_compatible, issues = compat.check_system_compatibility()
    
    if not is_compatible:
        print("❌ Compatibility issues found:")
        for issue in issues:
            print(f"  ⚠️ {issue}")
        print()
        
        install = input("Install missing dependencies? (y/N): ").strip().lower()
        if install == 'y':
            if compat.install_dependencies():
                print("✅ Dependencies installed - recheck compatibility")
                is_compatible, issues = compat.check_system_compatibility()
            else:
                print("❌ Dependency installation failed")
                return 1
    else:
        print("✅ System compatibility validated")
        
    # Validate SMBIOS access
    print("\n🔒 Validating SMBIOS access...")
    smbios_ok, smbios_issues = compat.validate_smbios_access()
    
    if not smbios_ok:
        print("⚠️ SMBIOS access issues:")
        for issue in smbios_issues:
            print(f"  ⚠️ {issue}")
    
    # Create distribution-specific files
    print("\n📝 Creating distribution-specific files...")
    
    # Create Makefile
    makefile_path = Path(f"Makefile.{compat.distro_info.name.lower()}")
    compat.create_distribution_makefile(makefile_path)
    
    # Create install script
    script_path = Path(f"install-{compat.distro_info.name.lower()}.sh")
    compat.create_install_script(script_path)
    
    print()
    print("🎉 Debian compatibility check complete!")
    
    if is_compatible:
        print("✅ System ready for SMBIOS token testing")
    else:
        print("⚠️ System needs additional configuration")
        
    return 0 if is_compatible else 1

if __name__ == "__main__":
    sys.exit(main())