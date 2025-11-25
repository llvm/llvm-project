#!/usr/bin/env python3
"""
MxGPU Virtualization Management Agent
KVM/Xen GPU passthrough, SR-IOV configuration, VM GPU allocation

Capabilities:
- GPU SR-IOV status and configuration
- KVM/Xen hypervisor GPU passthrough management
- Virtual function (VF) allocation and monitoring
- GPU scheduling and resource management
- FLR (Function Level Reset) management
- PF/VF handshake monitoring

Dependencies: libvirt (optional), subprocess for system commands
Note: Many operations require root/sudo privileges
"""

import os
import json
import subprocess
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MxGPUAgent:
    def __init__(self):
        """Initialize MxGPU virtualization agent"""
        self.data_dir = Path.home() / ".dsmil" / "mxgpu"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.configs_dir = self.data_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)

        self.logs_dir = self.data_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        # Track GPU configurations
        self.gpu_configs = {}
        self._load_configs()

        # Check system capabilities
        self.capabilities = self._check_capabilities()

    def _check_capabilities(self) -> Dict[str, bool]:
        """Check system capabilities"""
        caps = {}

        # Check if running on Linux
        caps['linux'] = os.name == 'posix' and 'linux' in os.uname().sysname.lower()

        # Check for AMD GPU
        try:
            result = subprocess.run(
                ['lspci', '-nn'],
                capture_output=True,
                text=True,
                timeout=5
            )
            caps['amd_gpu'] = 'AMD' in result.stdout and ('VGA' in result.stdout or 'Display' in result.stdout)
        except:
            caps['amd_gpu'] = False

        # Check for KVM
        try:
            result = subprocess.run(
                ['which', 'kvm'],
                capture_output=True,
                text=True,
                timeout=5
            )
            caps['kvm'] = result.returncode == 0
        except:
            caps['kvm'] = False

        # Check for QEMU
        try:
            result = subprocess.run(
                ['which', 'qemu-system-x86_64'],
                capture_output=True,
                text=True,
                timeout=5
            )
            caps['qemu'] = result.returncode == 0
        except:
            caps['qemu'] = False

        # Check for libvirt
        try:
            import libvirt
            caps['libvirt'] = True
        except ImportError:
            caps['libvirt'] = False

        # Check for IOMMU
        try:
            with open('/proc/cmdline', 'r') as f:
                cmdline = f.read()
                caps['iommu'] = 'iommu=on' in cmdline or 'amd_iommu=on' in cmdline or 'intel_iommu=on' in cmdline
        except:
            caps['iommu'] = False

        # Check for SR-IOV support
        try:
            sriov_path = Path('/sys/class/net')
            caps['sriov'] = any((p / 'device' / 'sriov_numvfs').exists() for p in sriov_path.iterdir() if p.is_dir())
        except:
            caps['sriov'] = False

        return caps

    def is_available(self) -> bool:
        """Check if agent can function"""
        return self.capabilities.get('linux', False)

    def _load_configs(self):
        """Load GPU configurations"""
        config_file = self.configs_dir / "gpu_configs.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.gpu_configs = json.load(f)

    def _save_configs(self):
        """Save GPU configurations"""
        config_file = self.configs_dir / "gpu_configs.json"
        with open(config_file, 'w') as f:
            json.dump(self.gpu_configs, f, indent=2)

    def detect_gpus(self) -> Dict[str, Any]:
        """
        Detect available GPUs and their SR-IOV capabilities

        Returns:
            Dict with GPU information
        """
        if not self.capabilities.get('linux'):
            return {
                "success": False,
                "error": "GPU detection requires Linux"
            }

        try:
            # Use lspci to detect GPUs
            result = subprocess.run(
                ['lspci', '-nn', '-k'],
                capture_output=True,
                text=True,
                timeout=10
            )

            gpus = []
            lines = result.stdout.split('\n')

            for i, line in enumerate(lines):
                if 'VGA' in line or 'Display' in line or '3D controller' in line:
                    # Extract PCI ID
                    pci_id = line.split()[0]

                    # Extract vendor and device
                    vendor = "Unknown"
                    device = "Unknown"

                    if 'AMD' in line or 'Advanced Micro Devices' in line:
                        vendor = "AMD"
                    elif 'NVIDIA' in line:
                        vendor = "NVIDIA"
                    elif 'Intel' in line:
                        vendor = "Intel"

                    # Extract device name
                    if '[' in line and ']' in line:
                        device = line.split('[')[1].split(']')[0]

                    # Check for SR-IOV capability
                    sriov_capable = False
                    try:
                        sriov_path = Path(f'/sys/bus/pci/devices/0000:{pci_id}/sriov_totalvfs')
                        if sriov_path.exists():
                            with open(sriov_path, 'r') as f:
                                total_vfs = int(f.read().strip())
                                sriov_capable = total_vfs > 0
                    except:
                        pass

                    gpus.append({
                        "pci_id": pci_id,
                        "vendor": vendor,
                        "device": device,
                        "sriov_capable": sriov_capable
                    })

            return {
                "success": True,
                "gpus": gpus,
                "count": len(gpus)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"GPU detection failed: {str(e)}"
            }

    def get_sriov_status(self, pci_id: str) -> Dict[str, Any]:
        """
        Get SR-IOV status for a GPU

        Args:
            pci_id: PCI ID (e.g., "01:00.0")

        Returns:
            Dict with SR-IOV status
        """
        try:
            base_path = Path(f'/sys/bus/pci/devices/0000:{pci_id}')

            if not base_path.exists():
                return {
                    "success": False,
                    "error": f"Device {pci_id} not found"
                }

            # Read total VFs
            total_vfs_path = base_path / 'sriov_totalvfs'
            if not total_vfs_path.exists():
                return {
                    "success": False,
                    "error": "SR-IOV not supported on this device"
                }

            with open(total_vfs_path, 'r') as f:
                total_vfs = int(f.read().strip())

            # Read current VFs
            num_vfs_path = base_path / 'sriov_numvfs'
            with open(num_vfs_path, 'r') as f:
                num_vfs = int(f.read().strip())

            return {
                "success": True,
                "pci_id": pci_id,
                "sriov_capable": True,
                "total_vfs": total_vfs,
                "active_vfs": num_vfs,
                "available_vfs": total_vfs - num_vfs
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get SR-IOV status: {str(e)}"
            }

    def enable_sriov(self, pci_id: str, num_vfs: int) -> Dict[str, Any]:
        """
        Enable SR-IOV on a GPU (requires root)

        Args:
            pci_id: PCI ID
            num_vfs: Number of virtual functions to create

        Returns:
            Dict with operation status
        """
        # This operation requires root - provide guidance
        command = f"echo {num_vfs} | sudo tee /sys/bus/pci/devices/0000:{pci_id}/sriov_numvfs"

        return {
            "success": False,
            "requires_root": True,
            "command": command,
            "message": "SR-IOV configuration requires root privileges. Run the command above manually.",
            "note": "Ensure IOMMU is enabled in BIOS and kernel parameters (amd_iommu=on or intel_iommu=on)"
        }

    def get_iommu_groups(self) -> Dict[str, Any]:
        """
        Get IOMMU groups for GPU passthrough

        Returns:
            Dict with IOMMU group information
        """
        try:
            iommu_path = Path('/sys/kernel/iommu_groups')

            if not iommu_path.exists():
                return {
                    "success": False,
                    "error": "IOMMU not enabled. Add amd_iommu=on or intel_iommu=on to kernel parameters."
                }

            groups = {}

            for group_path in iommu_path.iterdir():
                if not group_path.is_dir():
                    continue

                group_id = group_path.name
                devices_path = group_path / 'devices'

                if devices_path.exists():
                    devices = []
                    for device in devices_path.iterdir():
                        device_id = device.name
                        # Get device info from lspci
                        try:
                            result = subprocess.run(
                                ['lspci', '-s', device_id],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            device_info = result.stdout.strip()
                            devices.append({
                                "pci_id": device_id,
                                "description": device_info
                            })
                        except:
                            devices.append({"pci_id": device_id})

                    groups[group_id] = devices

            return {
                "success": True,
                "iommu_groups": groups,
                "count": len(groups)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get IOMMU groups: {str(e)}"
            }

    def generate_vm_config(self, vm_name: str,
                          gpu_pci_id: str,
                          vcpus: int = 4,
                          memory_gb: int = 8,
                          hypervisor: str = "kvm") -> Dict[str, Any]:
        """
        Generate VM configuration with GPU passthrough

        Args:
            vm_name: VM name
            gpu_pci_id: GPU PCI ID
            vcpus: Number of vCPUs
            memory_gb: Memory in GB
            hypervisor: 'kvm' or 'xen'

        Returns:
            Dict with configuration file path
        """
        if hypervisor == "kvm":
            config = self._generate_kvm_config(vm_name, gpu_pci_id, vcpus, memory_gb)
        elif hypervisor == "xen":
            config = self._generate_xen_config(vm_name, gpu_pci_id, vcpus, memory_gb)
        else:
            return {
                "success": False,
                "error": f"Unknown hypervisor: {hypervisor}"
            }

        # Save configuration
        config_file = self.configs_dir / f"{vm_name}_{hypervisor}.xml"
        with open(config_file, 'w') as f:
            f.write(config)

        return {
            "success": True,
            "vm_name": vm_name,
            "hypervisor": hypervisor,
            "config_file": str(config_file),
            "config": config
        }

    def _generate_kvm_config(self, vm_name: str, gpu_pci_id: str,
                            vcpus: int, memory_gb: int) -> str:
        """Generate KVM/QEMU libvirt XML configuration"""
        # Parse PCI ID
        pci_parts = gpu_pci_id.split(':')
        if len(pci_parts) >= 2:
            pci_bus = pci_parts[0]
            pci_slot = pci_parts[1].split('.')[0]
            pci_func = pci_parts[1].split('.')[1] if '.' in pci_parts[1] else '0'
        else:
            pci_bus, pci_slot, pci_func = "01", "00", "0"

        config = f'''<domain type='kvm'>
  <name>{vm_name}</name>
  <memory unit='GiB'>{memory_gb}</memory>
  <vcpu placement='static'>{vcpus}</vcpu>
  <os>
    <type arch='x86_64' machine='pc-q35-6.2'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
    <hyperv>
      <relaxed state='on'/>
      <vapic state='on'/>
      <spinlocks state='on' retries='8191'/>
    </hyperv>
    <kvm>
      <hidden state='on'/>
    </kvm>
  </features>
  <cpu mode='host-passthrough' check='none'/>
  <devices>
    <emulator>/usr/bin/qemu-system-x86_64</emulator>
    <!-- GPU Passthrough via MxGPU SR-IOV -->
    <hostdev mode='subsystem' type='pci' managed='yes'>
      <source>
        <address domain='0x0000' bus='0x{pci_bus}' slot='0x{pci_slot}' function='0x{pci_func}'/>
      </source>
    </hostdev>
    <!-- Virtual Display (for initial setup) -->
    <graphics type='vnc' port='-1' autoport='yes' listen='0.0.0.0'>
      <listen type='address' address='0.0.0.0'/>
    </graphics>
  </devices>
</domain>
'''
        return config

    def _generate_xen_config(self, vm_name: str, gpu_pci_id: str,
                            vcpus: int, memory_gb: int) -> str:
        """Generate Xen configuration"""
        config = f'''# Xen VM Configuration with GPU Passthrough
name = "{vm_name}"
type = "hvm"
memory = {memory_gb * 1024}
vcpus = {vcpus}
maxvcpus = {vcpus}

# GPU Passthrough
pci = [ '{gpu_pci_id}' ]

# Enable IOMMU passthrough
pci_permissive = 1

# Boot configuration
builder = "hvm"
boot = "c"

# VNC console
vnc = 1
vnclisten = "0.0.0.0"
vncpasswd = ""

# Device model
device_model_version = "qemu-xen"

# Disk (example - adjust as needed)
# disk = [ 'file:/path/to/disk.img,xvda,w' ]

# Network (example - adjust as needed)
# vif = [ 'bridge=xenbr0' ]
'''
        return config

    def check_vfio_status(self) -> Dict[str, Any]:
        """
        Check VFIO driver status (required for GPU passthrough)

        Returns:
            Dict with VFIO status
        """
        try:
            # Check if VFIO modules are loaded
            result = subprocess.run(
                ['lsmod'],
                capture_output=True,
                text=True,
                timeout=5
            )

            vfio_pci = 'vfio_pci' in result.stdout
            vfio = 'vfio' in result.stdout
            vfio_iommu = 'vfio_iommu_type1' in result.stdout

            return {
                "success": True,
                "vfio_loaded": vfio,
                "vfio_pci_loaded": vfio_pci,
                "vfio_iommu_loaded": vfio_iommu,
                "ready_for_passthrough": vfio and vfio_pci and vfio_iommu,
                "load_commands": [
                    "sudo modprobe vfio",
                    "sudo modprobe vfio_pci",
                    "sudo modprobe vfio_iommu_type1"
                ] if not (vfio and vfio_pci and vfio_iommu) else []
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to check VFIO status: {str(e)}"
            }

    def get_status(self) -> Dict[str, Any]:
        """Get agent status and system capabilities"""
        return {
            "available": self.is_available(),
            "capabilities": self.capabilities,
            "recommendations": self._get_recommendations(),
            "storage_path": str(self.data_dir)
        }

    def _get_recommendations(self) -> List[str]:
        """Generate setup recommendations based on system status"""
        recommendations = []

        if not self.capabilities.get('amd_gpu'):
            recommendations.append("No AMD GPU detected. MxGPU requires AMD GPUs with SR-IOV support.")

        if not self.capabilities.get('iommu'):
            recommendations.append("IOMMU not enabled. Add 'amd_iommu=on' or 'intel_iommu=on' to kernel boot parameters.")

        if not self.capabilities.get('kvm') and not self.capabilities.get('qemu'):
            recommendations.append("KVM/QEMU not installed. Install: sudo apt install qemu-kvm libvirt-daemon-system")

        if not self.capabilities.get('libvirt'):
            recommendations.append("Libvirt Python bindings not installed. Install: pip install libvirt-python")

        if not self.capabilities.get('sriov'):
            recommendations.append("SR-IOV support not detected. Ensure your GPU and motherboard support SR-IOV.")

        if not recommendations:
            recommendations.append("System appears ready for GPU virtualization!")

        return recommendations

# Export
__all__ = ['MxGPUAgent']
