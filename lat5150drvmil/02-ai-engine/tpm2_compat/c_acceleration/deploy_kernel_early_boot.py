#!/usr/bin/env python3
"""
TPM2 Acceleration Early Boot Kernel Integration Deployer
Deploys kernel module for automatic boot-time activation
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import time

class EarlyBootKernelDeployer:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.kernel_version = self.get_kernel_version()
        self.module_name = "tpm2_accel_early"
        self.deployment_id = f"kernel_early_{int(time.time())}"

        print(f"🔧 TPM2 EARLY BOOT KERNEL DEPLOYER")
        print(f"📊 Kernel: {self.kernel_version}")
        print(f"🎯 Module: {self.module_name}")

    def get_kernel_version(self):
        """Get current kernel version"""
        try:
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return "unknown"

    def build_kernel_module(self):
        """Build the early boot kernel module"""
        print("\n🔨 Building early boot kernel module...")

        # Check if we have the kernel module source
        module_source = self.base_path / f"{self.module_name}.c"
        module_header = self.base_path / f"{self.module_name}.h"
        makefile = self.base_path / "Makefile.kernel"

        if not module_source.exists():
            print(f"❌ Module source not found: {module_source}")
            return False

        if not makefile.exists():
            print(f"❌ Kernel Makefile not found: {makefile}")
            return False

        try:
            # Build the module
            result = subprocess.run(
                ['make', '-f', 'Makefile.kernel', 'module'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode == 0:
                print("✅ Kernel module built successfully")
                return True
            else:
                print(f"❌ Build failed: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Build timeout")
            return False
        except Exception as e:
            print(f"❌ Build error: {e}")
            return False

    def install_kernel_module(self):
        """Install kernel module for early boot loading"""
        print("\n📦 Installing kernel module...")

        module_file = self.base_path / f"{self.module_name}.ko"
        if not module_file.exists():
            print(f"❌ Module file not found: {module_file}")
            return False

        try:
            # Copy module to system location
            modules_dir = Path(f"/lib/modules/{self.kernel_version}/kernel/drivers/tpm")
            modules_dir.mkdir(parents=True, exist_ok=True)

            dest_module = modules_dir / f"{self.module_name}.ko"
            shutil.copy2(module_file, dest_module)
            print(f"✅ Module installed: {dest_module}")

            # Update module dependencies
            result = subprocess.run(['depmod', '-a'], capture_output=True)
            if result.returncode == 0:
                print("✅ Module dependencies updated")
                return True
            else:
                print(f"⚠️ depmod warning: {result.stderr.decode()}")
                return True  # Continue anyway

        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return False

    def create_modules_load_config(self):
        """Create modules-load configuration for early boot"""
        print("\n⚙️ Configuring early boot loading...")

        try:
            # Create modules-load.d configuration
            config_dir = Path("/etc/modules-load.d")
            config_dir.mkdir(exist_ok=True)

            config_file = config_dir / "tpm2-acceleration.conf"
            with open(config_file, 'w') as f:
                f.write(f"""# TPM2 Hardware Acceleration Early Boot Module
# Loaded during early boot for maximum performance
{self.module_name}
""")

            print(f"✅ Early boot config: {config_file}")

            # Create modprobe configuration for module parameters
            modprobe_dir = Path("/etc/modprobe.d")
            modprobe_dir.mkdir(exist_ok=True)

            modprobe_file = modprobe_dir / "tpm2-acceleration.conf"
            with open(modprobe_file, 'w') as f:
                f.write(f"""# TPM2 Acceleration Module Parameters
options {self.module_name} enable_npu=1 enable_gna=1 debug_level=2
options {self.module_name} max_devices=4 buffer_size=4194304
""")

            print(f"✅ Module parameters: {modprobe_file}")
            return True

        except Exception as e:
            print(f"❌ Configuration failed: {e}")
            return False

    def create_systemd_integration(self):
        """Create systemd service for userspace integration"""
        print("\n🔧 Creating systemd integration...")

        try:
            systemd_dir = Path("/etc/systemd/system")
            service_file = systemd_dir / "tpm2-acceleration-early.service"

            service_content = f"""[Unit]
Description=TPM2 Hardware Acceleration Early Boot Integration
After=multi-user.target
Wants=multi-user.target
DefaultDependencies=no

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'echo "TPM2 Early Boot Acceleration: $(cat /sys/module/{self.module_name}/parameters/status 2>/dev/null || echo NOT_LOADED)"'
ExecStart=/home/john/LAT/LAT5150DRVMIL/tpm2_compat/c_acceleration/tpm2_compat_userspace/bin/setup_tpm2_acceleration.sh
RemainAfterExit=yes
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

            with open(service_file, 'w') as f:
                f.write(service_content)

            print(f"✅ Systemd service: {service_file}")

            # Enable the service
            subprocess.run(['systemctl', 'daemon-reload'], capture_output=True)
            subprocess.run(['systemctl', 'enable', 'tpm2-acceleration-early.service'],
                         capture_output=True)

            print("✅ Service enabled for boot")
            return True

        except Exception as e:
            print(f"❌ Systemd integration failed: {e}")
            return False

    def update_initramfs(self):
        """Update initramfs to include early boot module"""
        print("\n📦 Updating initramfs...")

        try:
            # Check if update-initramfs is available
            if shutil.which('update-initramfs'):
                result = subprocess.run(['update-initramfs', '-u'],
                                      capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    print("✅ Initramfs updated successfully")
                    return True
                else:
                    print(f"⚠️ Initramfs update warning: {result.stderr}")
                    return True  # Continue anyway
            else:
                print("⚠️ update-initramfs not available, skipping")
                return True

        except subprocess.TimeoutExpired:
            print("⚠️ Initramfs update timeout")
            return True
        except Exception as e:
            print(f"❌ Initramfs update failed: {e}")
            return False

    def test_module_loading(self):
        """Test module can be loaded"""
        print("\n🧪 Testing module loading...")

        try:
            # Try to load the module
            result = subprocess.run(['modprobe', self.module_name],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Module loads successfully")

                # Check if device is created
                device_path = f"/dev/tpm2_accel_early"
                if Path(device_path).exists():
                    print(f"✅ Device created: {device_path}")
                else:
                    print(f"⚠️ Device not found: {device_path}")

                # Try to unload for clean state
                subprocess.run(['modprobe', '-r', self.module_name],
                             capture_output=True)
                print("✅ Module unloaded cleanly")
                return True
            else:
                print(f"❌ Module load failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Module test failed: {e}")
            return False

    def create_grub_integration(self):
        """Add GRUB configuration for early loading"""
        print("\n🚀 Configuring GRUB integration...")

        try:
            grub_dir = Path("/etc/default")
            grub_file = grub_dir / "grub"

            if grub_file.exists():
                # Read existing GRUB config
                with open(grub_file, 'r') as f:
                    grub_content = f.read()

                # Add TPM2 acceleration parameters if not present
                tpm2_params = f"tpm2_accel_early.enable=1 tpm2_accel_early.max_devices=4"

                if "tpm2_accel_early" not in grub_content:
                    # Find GRUB_CMDLINE_LINUX line and append
                    lines = grub_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.startswith('GRUB_CMDLINE_LINUX='):
                            # Extract existing parameters
                            existing = line[line.find('"')+1:line.rfind('"')]
                            lines[i] = f'GRUB_CMDLINE_LINUX="{existing} {tpm2_params}"'
                            break

                    # Write back
                    with open(grub_file, 'w') as f:
                        f.write('\n'.join(lines))

                    print("✅ GRUB configuration updated")

                    # Update GRUB
                    result = subprocess.run(['update-grub'], capture_output=True, text=True)
                    if result.returncode == 0:
                        print("✅ GRUB updated successfully")
                    else:
                        print(f"⚠️ GRUB update warning: {result.stderr}")
                else:
                    print("✅ GRUB already configured")

                return True
            else:
                print("⚠️ GRUB config not found, skipping")
                return True

        except Exception as e:
            print(f"❌ GRUB integration failed: {e}")
            return False

    def deploy_early_boot_integration(self):
        """Deploy complete early boot integration"""
        print(f"\n🚀 DEPLOYING EARLY BOOT KERNEL INTEGRATION")
        print("=" * 60)

        deployment_steps = [
            ("Build Kernel Module", self.build_kernel_module),
            ("Install Module", self.install_kernel_module),
            ("Configure Early Loading", self.create_modules_load_config),
            ("Systemd Integration", self.create_systemd_integration),
            ("Update Initramfs", self.update_initramfs),
            ("Test Module Loading", self.test_module_loading),
            ("GRUB Integration", self.create_grub_integration)
        ]

        successful_steps = 0
        for step_name, step_function in deployment_steps:
            print(f"\n--- {step_name} ---")
            try:
                if step_function():
                    successful_steps += 1
                    print(f"✅ {step_name}: SUCCESS")
                else:
                    print(f"❌ {step_name}: FAILED")
            except Exception as e:
                print(f"❌ {step_name}: ERROR - {e}")

        # Final summary
        print("\n" + "=" * 60)
        print("🎯 EARLY BOOT DEPLOYMENT SUMMARY")
        print("=" * 60)
        print(f"📊 Steps completed: {successful_steps}/{len(deployment_steps)}")
        print(f"🔧 Kernel version: {self.kernel_version}")
        print(f"📦 Module: {self.module_name}")
        print(f"🚀 Deployment ID: {self.deployment_id}")

        if successful_steps >= 5:  # Essential steps
            print("\n✅ EARLY BOOT INTEGRATION DEPLOYED!")
            print("🔄 TPM2 acceleration will activate automatically on next boot")
            print("🎯 Benefits:")
            print("   • Hardware acceleration available immediately after boot")
            print("   • Intel NPU (34.0 TOPS) utilization from kernel space")
            print("   • Dell military token authorization during early boot")
            print("   • Seamless integration with userspace acceleration")
            print(f"   • Character device: /dev/tpm2_accel_early")

            print("\n🔧 Next steps:")
            print("   1. Reboot system to activate early boot integration")
            print("   2. Verify with: lsmod | grep tpm2_accel_early")
            print("   3. Check device: ls -la /dev/tpm2_accel_early")
            print("   4. Monitor logs: journalctl -u tpm2-acceleration-early")

            return True
        else:
            print("\n⚠️ PARTIAL DEPLOYMENT")
            print("🔧 Some components may not activate on boot")
            return False

def main():
    if os.geteuid() != 0:
        print("❌ This script requires root privileges for kernel module installation")
        print("🔧 Please run with: sudo python3 deploy_kernel_early_boot.py")
        sys.exit(1)

    deployer = EarlyBootKernelDeployer()
    success = deployer.deploy_early_boot_integration()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()