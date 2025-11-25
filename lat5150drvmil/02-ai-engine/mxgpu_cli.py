#!/usr/bin/env python3
"""
MxGPU Virtualization CLI - Natural Language Interface
KVM/Xen GPU passthrough, SR-IOV configuration, VM GPU allocation

Usage:
    python3 mxgpu_cli.py "detect gpus"
    python3 mxgpu_cli.py "check sriov status for 01:00.0"
    python3 mxgpu_cli.py "generate kvm config for gaming-vm with gpu 01:00.0"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.mxgpu_wrapper import MxGPUAgent

class MxGPUCLI:
    def __init__(self):
        self.agent = MxGPUAgent()

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Detect GPUs
        if 'detect' in query_lower and 'gpu' in query_lower:
            return {'action': 'detect_gpus'}

        # SR-IOV status
        elif 'sriov' in query_lower and ('status' in query_lower or 'check' in query_lower):
            pci_match = re.search(r'(\d{2}:\d{2}\.\d)', query)
            pci_id = pci_match.group(1) if pci_match else None

            return {
                'action': 'sriov_status',
                'pci_id': pci_id
            }

        # IOMMU groups
        elif 'iommu' in query_lower or 'passthrough' in query_lower:
            return {'action': 'iommu_groups'}

        # Generate VM config
        elif 'generate' in query_lower or 'create' in query_lower or 'config' in query_lower:
            vm_match = re.search(r'(?:for |vm |name )([a-z0-9_-]+)', query_lower)
            vm_name = vm_match.group(1) if vm_match else None

            pci_match = re.search(r'(?:gpu |pci )(\d{2}:\d{2}\.\d)', query)
            gpu_pci_id = pci_match.group(1) if pci_match else None

            vcpu_match = re.search(r'(\d+)\s*(?:vcpu|cpu|core)', query_lower)
            vcpus = int(vcpu_match.group(1)) if vcpu_match else 4

            mem_match = re.search(r'(\d+)\s*(?:gb|gib)', query_lower)
            memory_gb = int(mem_match.group(1)) if mem_match else 8

            hypervisor = 'kvm'  # default
            if 'xen' in query_lower:
                hypervisor = 'xen'

            return {
                'action': 'generate_config',
                'vm_name': vm_name,
                'gpu_pci_id': gpu_pci_id,
                'vcpus': vcpus,
                'memory_gb': memory_gb,
                'hypervisor': hypervisor
            }

        # VFIO status
        elif 'vfio' in query_lower:
            return {'action': 'vfio_status'}

        # Status
        elif 'status' in query_lower or 'info' in query_lower:
            return {'action': 'status'}

        else:
            return {'action': 'help'}

    def execute(self, query: str):
        """Execute natural language command"""
        parsed = self.parse_command(query)
        action = parsed.get('action')

        if action == 'detect_gpus':
            result = self.agent.detect_gpus()

            if result.get('success'):
                print(f"✅ GPU detection complete!")
                print(f"   Found: {result['count']} GPU(s)")

                for gpu in result['gpus']:
                    print(f"\n   [{gpu['pci_id']}]")
                    print(f"      Vendor: {gpu['vendor']}")
                    print(f"      Device: {gpu['device']}")
                    print(f"      SR-IOV: {'✅ Supported' if gpu['sriov_capable'] else '❌ Not supported'}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'sriov_status':
            if not parsed.get('pci_id'):
                print("❌ Please specify a PCI ID (e.g., 01:00.0)")
                return

            result = self.agent.get_sriov_status(pci_id=parsed['pci_id'])

            if result.get('success'):
                print(f"✅ SR-IOV Status for {parsed['pci_id']}")
                print(f"   Capable: {result['sriov_capable']}")
                print(f"   Total VFs: {result['total_vfs']}")
                print(f"   Active VFs: {result['active_vfs']}")
                print(f"   Available VFs: {result['available_vfs']}")

                if result['active_vfs'] == 0:
                    print(f"\n   ℹ️  To enable SR-IOV:")
                    print(f"      echo {result['total_vfs']} | sudo tee /sys/bus/pci/devices/0000:{parsed['pci_id']}/sriov_numvfs")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'iommu_groups':
            result = self.agent.get_iommu_groups()

            if result.get('success'):
                print(f"✅ IOMMU Groups found: {result['count']}")

                for group_id, devices in list(result['iommu_groups'].items())[:10]:
                    print(f"\n   Group {group_id}:")
                    for device in devices:
                        print(f"      {device['pci_id']}: {device.get('description', 'Unknown device')}")

                if result['count'] > 10:
                    print(f"\n   ... and {result['count'] - 10} more groups")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'generate_config':
            if not parsed.get('vm_name') or not parsed.get('gpu_pci_id'):
                print("❌ Please specify VM name and GPU PCI ID")
                print("   Example: python3 mxgpu_cli.py \"generate kvm config for my-vm with gpu 01:00.0\"")
                return

            result = self.agent.generate_vm_config(
                vm_name=parsed['vm_name'],
                gpu_pci_id=parsed['gpu_pci_id'],
                vcpus=parsed['vcpus'],
                memory_gb=parsed['memory_gb'],
                hypervisor=parsed['hypervisor']
            )

            if result.get('success'):
                print(f"✅ VM configuration generated!")
                print(f"   VM Name: {result['vm_name']}")
                print(f"   Hypervisor: {result['hypervisor'].upper()}")
                print(f"   Config file: {result['config_file']}")
                print(f"\n   Configuration:")
                print("   " + "-" * 60)
                for line in result['config'].split('\n')[:20]:
                    print(f"   {line}")
                if result['config'].count('\n') > 20:
                    print(f"   ... (see full config in {result['config_file']})")
                print("   " + "-" * 60)
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'vfio_status':
            result = self.agent.check_vfio_status()

            if result.get('success'):
                print(f"✅ VFIO Status")
                print(f"   VFIO loaded: {'✅' if result['vfio_loaded'] else '❌'}")
                print(f"   VFIO-PCI loaded: {'✅' if result['vfio_pci_loaded'] else '❌'}")
                print(f"   VFIO-IOMMU loaded: {'✅' if result['vfio_iommu_loaded'] else '❌'}")
                print(f"   Ready for passthrough: {'✅' if result['ready_for_passthrough'] else '❌'}")

                if not result['ready_for_passthrough'] and result.get('load_commands'):
                    print(f"\n   ℹ️  To enable VFIO:")
                    for cmd in result['load_commands']:
                        print(f"      {cmd}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'status':
            status = self.agent.get_status()
            print(f"🖥️  MxGPU Agent Status")
            print(f"   Available: {status['available']}")

            print(f"\n   Capabilities:")
            for cap, avail in status['capabilities'].items():
                print(f"      {cap}: {'✅' if avail else '❌'}")

            if status.get('recommendations'):
                print(f"\n   Recommendations:")
                for rec in status['recommendations']:
                    print(f"      • {rec}")

            print(f"\n   Storage: {status['storage_path']}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
🖥️  MxGPU Virtualization CLI - Natural Language Interface

KVM/Xen GPU Passthrough, SR-IOV Configuration, VM GPU Allocation

USAGE:
    python3 mxgpu_cli.py "your natural language command"

EXAMPLES:
    # Detection
    python3 mxgpu_cli.py "detect gpus"
    python3 mxgpu_cli.py "check sriov status for 01:00.0"
    python3 mxgpu_cli.py "show iommu groups"

    # Configuration
    python3 mxgpu_cli.py "generate kvm config for gaming-vm with gpu 01:00.0"
    python3 mxgpu_cli.py "create xen config for render-vm with gpu 01:00.0 8 vcpus 16gb"

    # Status
    python3 mxgpu_cli.py "check vfio status"
    python3 mxgpu_cli.py "status"

PREREQUISITES:
    1. Linux system with IOMMU support
    2. AMD/NVIDIA/Intel GPU with SR-IOV
    3. IOMMU enabled in BIOS and kernel:
       - Add to kernel boot: amd_iommu=on (AMD) or intel_iommu=on (Intel)
    4. VFIO drivers loaded (optional for status check)

TEMPEST COMPLIANCE:
    - All operations local (no network required)
    - Air-gapped deployment compatible
    - Suitable for classified computing environments
    - GPU virtualization for secure workloads
    - EM emissions: GPU-dependent, consult vendor specs
    - Recommend: Faraday cage for classified GPU workloads

SECURITY NOTES:
    - GPU passthrough provides hardware-level isolation
    - Suitable for multi-level security (MLS) deployments
    - SR-IOV enables secure multi-tenancy
    - IOMMU provides DMA attack protection
        """)

def main():
    if len(sys.argv) < 2:
        cli = MxGPUCLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = MxGPUCLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
