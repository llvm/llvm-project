# LAT5150 Xen Hypervisor Orchestration

**On-demand VM provisioning with full LAT5150 integration**

## Overview

The Xen Hypervisor Orchestrator provides on-demand virtual machine provisioning with deep integration into the LAT5150 platform. VMs can leverage Intel NPU, iGPU, and NCS2 accelerators through hardware passthrough, TPM 2.0 attestation for trusted boot, and quantum crypto for encryption.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Tactical Web UI (TEMPEST-Compliant)                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Xen Management Page                                    │ │
│  │  • VM lifecycle (create/start/stop)                     │ │
│  │  • Hardware passthrough config                          │ │
│  │  • Real-time status monitoring                          │ │
│  │  • Security classification banners                      │ │
│  └───────────────────┬────────────────────────────────────┘ │
└────────────────────┬─┴────────────────────────────────────────┘
                     │ REST API
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Xen Orchestrator (xen_orchestrator.py)                      │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │  VM Lifecycle  │  │   Hardware     │  │  Integration  │ │
│  │  Management    │  │   Passthrough  │  │   Layer       │ │
│  │                │  │                │  │               │ │
│  │  • Create      │  │  • NPU         │  │  • DSMIL      │ │
│  │  • Start       │  │  • iGPU        │  │  • Intel HW   │ │
│  │  • Stop        │  │  • NCS2        │  │  • TPM 2.0    │ │
│  │  • Status      │  │  • PCI         │  │  • Crypto     │ │
│  └────────────────┘  └────────────────┘  └───────────────┘ │
└────────────────────┬─────────────────────────────────────────┘
                     │ xl commands
                     ▼
┌──────────────────────────────────────────────────────────────┐
│  Xen Hypervisor (xl toolstack)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  VM 1 (CUI)  │  │  VM 2        │  │  VM 3        │       │
│  │  NPU Pass    │  │  iGPU Pass   │  │  NCS2 Pass   │       │
│  │  4vCPU       │  │  8vCPU       │  │  16vCPU      │       │
│  │  8GB RAM     │  │  16GB RAM    │  │  32GB RAM    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

## File Structure

```
04-hypervisor/
├── xen_orchestrator.py          # Core orchestration engine (800 lines)
│   ├── XenOrchestrator          # Main orchestrator class
│   ├── VMConfig                 # VM configuration dataclass
│   ├── VMStatus                 # VM status tracking
│   ├── PCIDevice                # PCI passthrough support
│   └── Integration with:
│       ├── Intel hardware discovery
│       ├── DSMIL subsystem controller
│       ├── TPM 2.0 attestation
│       └── Quantum crypto layer
│
├── register_xen_page.py         # Page registration for web UI
│   ├── TEMPEST-compliant HTML
│   ├── Tactical theme integration
│   └── Real-time VM monitoring
│
├── xen_api_handlers.py          # Flask API endpoints
│   ├── /api/hypervisor/status
│   ├── /api/hypervisor/vms
│   ├── /api/hypervisor/create-vm
│   ├── /api/hypervisor/start-vm
│   ├── /api/hypervisor/stop-vm
│   ├── /api/hypervisor/ai-acceleration
│   └── /api/hypervisor/pci-devices
│
└── README.md                    # This file

Generated at runtime:
├── /opt/lat5150/hypervisor/configs/    # VM configurations (.cfg, .json)
└── /opt/lat5150/hypervisor/storage/    # VM disk images
```

## Features

### ✅ **On-Demand VM Provisioning**

Create VMs programmatically or via web UI:

```python
from xen_orchestrator import get_xen_orchestrator, VMConfig, VMClassification

orchestrator = get_xen_orchestrator()

config = VMConfig(
    name="analysis-vm-01",
    uuid="...",
    vcpus=4,
    memory_mb=4096,
    classification=VMClassification.CUI,
    disk_path="/opt/lat5150/hypervisor/storage/analysis-01.img",
    tpm_enabled=True,
    secure_boot=True,
    npu_passthrough=True
)

orchestrator.create_vm(config)
orchestrator.start_vm("analysis-vm-01")
```

### ✅ **AI Acceleration Passthrough**

**100 TOPS Available for VMs:**

| Component | TOPS | Passthrough Method |
|-----------|------|-------------------|
| Intel NPU | 30 TOPS | PCIe device (/dev/accel/accel0) |
| Intel iGPU | 40 TOPS | PCI passthrough (full GPU) |
| Intel NCS2 | 30 TOPS | USB passthrough (per stick) |

**Example: NPU Passthrough**

```python
config = VMConfig(
    name="npu-inference-vm",
    vcpus=8,
    memory_mb=16384,
    classification=VMClassification.SECRET,
    disk_path="/opt/lat5150/hypervisor/storage/npu-vm.img",
    npu_passthrough=True,  # Passthrough Intel NPU (30 TOPS)
    tpm_enabled=True
)

orchestrator.create_vm(config)
```

**Generated Xen Config:**
```
pci = ['0000:00:0b.0']  # NPU device
```

### ✅ **DSMIL Device Integration**

Pass DSMIL devices to VMs for hardware reconnaissance:

```python
config.dsmil_devices = [0x8001, 0x8002, 0x8003]  # Specific DSMIL device IDs
```

### ✅ **Security Classification**

VMs inherit security controls based on classification:

| Classification | Features |
|----------------|----------|
| UNCLASSIFIED | Standard isolation |
| CUI | TPM 2.0 recommended |
| SECRET | TPM 2.0 + Secure Boot required |
| TOP SECRET | TPM 2.0 + Secure Boot + Encryption required |

### ✅ **TPM 2.0 Integration**

Trusted boot with hardware attestation:

```python
config.tpm_enabled = True
config.secure_boot = True
```

Generated Xen config:
```
vtpm = ['backend=0']
secureboot = 1
```

### ✅ **Quantum Crypto Integration**

VM disk encryption via quantum crypto layer:

```python
config.encrypted = True
# Uses CSNA 2.0 compliant encryption
```

### ✅ **TEMPEST-Compliant Management**

Web UI follows tactical design system:
- Three display modes (Comfort/Day/Night)
- No animations (EMF reduction)
- Security classification banners
- Real-time status updates

## Quick Start

### 1. Register the Management Page

```bash
cd /home/user/LAT5150DRVMIL/04-hypervisor
python3 register_xen_page.py
```

**Output:**
```
======================================================================
Registering Xen Hypervisor Management Page
======================================================================

✓ Successfully registered: Xen Hypervisor Management
  Page ID: xen_hypervisor
  Route: /hypervisor
  Endpoints: 7
  TEMPEST Compliant: True

Access at: http://localhost:5001/page/xen_hypervisor
```

### 2. Integrate with Flask App

```python
from flask import Flask
from xen_api_handlers import register_xen_api_handlers
from dynamic_page_api import register_page_api_routes

app = Flask(__name__)

# Register page management APIs
register_page_api_routes(app)

# Register Xen API handlers
register_xen_api_handlers(app)

app.run(host='127.0.0.1', port=5001)
```

### 3. Access Web UI

Navigate to: `http://localhost:5001/page/xen_hypervisor`

## Usage Examples

### Example 1: Create Analysis VM with NPU

```python
from xen_orchestrator import get_xen_orchestrator, VMConfig, VMClassification
import uuid

orchestrator = get_xen_orchestrator()

config = VMConfig(
    name="threat-analysis-01",
    uuid=str(uuid.uuid4()),
    vcpus=8,
    memory_mb=16384,
    classification=VMClassification.CUI,
    disk_path="/opt/lat5150/hypervisor/storage/threat-analysis-01.img",
    network_bridge="xenbr0",
    tpm_enabled=True,
    secure_boot=True,
    encrypted=False,
    npu_passthrough=True  # Intel NPU (30 TOPS)
)

# Create and start
orchestrator.create_vm(config)
orchestrator.start_vm("threat-analysis-01")

# Check status
status = orchestrator.get_vm_status("threat-analysis-01")
print(f"VM State: {status.state.value}")
print(f"Uptime: {status.uptime_seconds}s")
```

### Example 2: GPU Passthrough for ML Training

```python
config = VMConfig(
    name="ml-training-vm",
    uuid=str(uuid.uuid4()),
    vcpus=16,
    memory_mb=32768,
    classification=VMClassification.CUI,
    disk_path="/opt/lat5150/hypervisor/storage/ml-training.img",
    gpu_passthrough=True,  # Intel iGPU (40 TOPS + full UMA)
    tpm_enabled=True
)

orchestrator.create_vm(config)
orchestrator.start_vm("ml-training-vm")
```

### Example 3: NCS2 Cluster for Distributed Inference

```python
config = VMConfig(
    name="inference-cluster-01",
    uuid=str(uuid.uuid4()),
    vcpus=4,
    memory_mb=8192,
    classification=VMClassification.UNCLASS,
    disk_path="/opt/lat5150/hypervisor/storage/inference-01.img",
    ncs2_devices=[0, 1, 2]  # All 3 NCS2 sticks (30 TOPS total)
)

orchestrator.create_vm(config)
orchestrator.start_vm("inference-cluster-01")
```

### Example 4: Secure Boot SECRET VM

```python
config = VMConfig(
    name="classified-ops-vm",
    uuid=str(uuid.uuid4()),
    vcpus=12,
    memory_mb=24576,
    classification=VMClassification.SECRET,
    disk_path="/opt/lat5150/hypervisor/storage/classified-ops.img",
    tpm_enabled=True,       # Required for SECRET
    secure_boot=True,       # Required for SECRET
    encrypted=True,         # Quantum crypto
    npu_passthrough=False,  # No hardware passthrough for classified
    gpu_passthrough=False
)

orchestrator.create_vm(config)
orchestrator.start_vm("classified-ops-vm")
```

## API Reference

### Python API

#### `XenOrchestrator`

```python
from xen_orchestrator import get_xen_orchestrator

orchestrator = get_xen_orchestrator()
```

**Methods:**

- `create_vm(config: VMConfig) -> bool` - Create and register VM
- `start_vm(name: str) -> bool` - Start VM
- `stop_vm(name: str, force: bool = False) -> bool` - Stop VM
- `get_vm_status(name: str) -> Optional[VMStatus]` - Get VM status
- `list_vms() -> List[VMStatus]` - List all VMs
- `get_available_pci_devices() -> List[PCIDevice]` - Get PCI devices
- `get_ai_acceleration_summary() -> Dict` - Get AI acceleration info

### REST API

All endpoints require authentication and are rate-limited.

#### `GET /api/hypervisor/status`

Get orchestrator status.

**Response:**
```json
{
    "registered_vms": 3,
    "running_vms": 2,
    "intel_platform_available": true,
    "dsmil_available": true,
    "total_ai_tops": 100.0
}
```

#### `GET /api/hypervisor/vms`

List all VMs.

**Response:**
```json
{
    "vms": [
        {
            "name": "analysis-vm-01",
            "state": "running",
            "uuid": "...",
            "vcpus": 4,
            "memory_mb": 4096,
            "uptime_seconds": 3600
        }
    ]
}
```

#### `POST /api/hypervisor/create-vm`

Create a new VM.

**Request:**
```json
{
    "name": "test-vm",
    "vcpus": 4,
    "memory_mb": 4096,
    "classification": "CUI",
    "disk_path": "/opt/lat5150/hypervisor/storage/test.img",
    "tpm_enabled": true,
    "npu_passthrough": true
}
```

**Response:**
```json
{
    "success": true,
    "name": "test-vm",
    "uuid": "..."
}
```

#### `POST /api/hypervisor/start-vm`

Start a VM.

**Request:**
```json
{
    "name": "test-vm"
}
```

#### `POST /api/hypervisor/stop-vm`

Stop a VM.

**Request:**
```json
{
    "name": "test-vm",
    "force": false
}
```

#### `GET /api/hypervisor/ai-acceleration`

Get AI acceleration summary.

**Response:**
```json
{
    "total_tops": 100.0,
    "components": [
        {
            "type": "NPU",
            "model": "Intel AI Boost NPU",
            "tops": 30.0,
            "passthrough_available": true
        },
        {
            "type": "iGPU",
            "model": "Intel Arc Graphics",
            "tops": 40.0,
            "passthrough_available": true
        },
        {
            "type": "NCS2",
            "count": 3,
            "total_tops": 30.0,
            "passthrough_available": true
        }
    ]
}
```

#### `GET /api/hypervisor/pci-devices`

Get available PCI devices.

**Response:**
```json
{
    "devices": [
        {
            "domain": "0000",
            "bus": "00",
            "slot": "02",
            "function": "0",
            "device_id": "7d55",
            "vendor_id": "8086",
            "description": "VGA compatible controller: Intel Corporation"
        }
    ]
}
```

## Integration Points

### Intel Hardware Detection

Automatic detection of AI accelerators:

```python
orchestrator.intel_platform  # IntelPlatform object
# - platform.npu: NPU information
# - platform.gpu: iGPU information
# - platform.ncs2: NCS2 information
# - platform.total_ai_tops: Total TOPS
```

### DSMIL Subsystem Controller

Access to 84 DSMIL devices:

```python
orchestrator.dsmil_controller  # DSMILSubsystemController
# - Safe devices: 79
# - Quarantined: 5
```

### TPM 2.0 Attestation

Trusted boot integration (requires `vtpm` support):

```python
config.tpm_enabled = True
# VM will have virtual TPM attached
# Can attest boot chain via TPM 2.0
```

### Quantum Crypto Layer

CSNA 2.0 compliant encryption:

```python
config.encrypted = True
# VM disk encrypted with quantum-resistant algorithms
```

## Configuration

### Directory Structure

```bash
/opt/lat5150/hypervisor/
├── configs/                      # VM configurations
│   ├── analysis-vm-01.cfg       # Xen config file
│   ├── analysis-vm-01.json      # Metadata
│   └── ...
├── storage/                      # VM disk images
│   ├── analysis-vm-01.img
│   └── ...
└── logs/                         # Operation logs
    └── orchestrator.log
```

### Xen Configuration

VMs use Xen's `xl` toolstack. Example generated config:

```
# Xen VM Configuration: analysis-vm-01
# Classification: CUI

name = 'analysis-vm-01'
uuid = '...'

# Resources
vcpus = 4
memory = 4096

# Boot
kernel = '/boot/vmlinuz'
ramdisk = '/boot/initrd.img'
root = '/dev/xvda1 ro'

# Disk
disk = ['file:/opt/lat5150/hypervisor/storage/analysis-vm-01.img,xvda,w']

# Network
vif = ['bridge=xenbr0']

# Hardware Passthrough
pci = ['0000:00:0b.0']  # NPU

# Security
vtpm = ['backend=0']
secureboot = 1
```

## Security

### Localhost-Only

All APIs are localhost-only with APT-grade security:
- 127.0.0.1 binding
- Token authentication
- Rate limiting
- Input validation
- Audit logging

### VM Isolation

VMs run with Xen's hardware isolation:
- Separate memory spaces
- CPU scheduling isolation
- Network isolation (separate bridges)
- Storage isolation (separate disk images)

### Classification Enforcement

Security features enforced by classification:

```python
if config.classification in [VMClassification.SECRET, VMClassification.TOP_SECRET]:
    if not config.tpm_enabled or not config.secure_boot:
        raise ValueError("SECRET/TS VMs require TPM + Secure Boot")
```

## Troubleshooting

### "xl: command not found"

Install Xen tools:

```bash
sudo apt-get install xen-tools xen-utils-common
```

### "Failed to create VM"

Check Xen hypervisor is running:

```bash
xl info
xl list
```

### "PCI passthrough not working"

Enable IOMMU in GRUB:

```bash
# Edit /etc/default/grub
GRUB_CMDLINE_LINUX_DEFAULT="intel_iommu=on iommu=pt"

# Update GRUB
sudo update-grub
sudo reboot
```

### "NPU not available"

Check NPU device:

```bash
ls -l /dev/accel/accel0
# Should show NPU device

# Check with xl
xl pci-list-assignable-devices
```

## Performance

### VM Startup Time

- Without passthrough: ~2-5 seconds
- With PCI passthrough: ~5-10 seconds
- With TPM/Secure Boot: ~10-15 seconds

### Resource Overhead

- Xen hypervisor: ~256MB RAM
- Domain-0 (management): ~512MB RAM
- Per VM overhead: ~128MB RAM

### AI Acceleration

VMs with passthrough get full hardware performance:

- NPU: 30 TOPS (no overhead)
- iGPU: 40 TOPS (no overhead)
- NCS2: 10 TOPS per stick (no overhead)

## Best Practices

1. **Use TPM for Sensitive VMs**
   ```python
   if classification in [CUI, SECRET, TOP_SECRET]:
       config.tpm_enabled = True
   ```

2. **Allocate Resources Wisely**
   ```python
   # Leave resources for Domain-0
   max_vcpus = total_cpus - 2
   max_memory = total_memory - 2048
   ```

3. **Use Proper Disk Paths**
   ```python
   disk_path = f"/opt/lat5150/hypervisor/storage/{vm_name}.img"
   ```

4. **Monitor VM Status**
   ```python
   status = orchestrator.get_vm_status(name)
   if status.state != VMState.RUNNING:
       # Handle error
   ```

5. **Graceful Shutdown**
   ```python
   orchestrator.stop_vm(name, force=False)  # Graceful
   # Wait for shutdown
   orchestrator.stop_vm(name, force=True)   # Force if needed
   ```

---

## Version History

- **v1.0.0** (2025-01-13)
  - Initial release
  - VM lifecycle management
  - AI acceleration passthrough (NPU/iGPU/NCS2)
  - DSMIL integration
  - TPM 2.0 support
  - TEMPEST-compliant web UI
  - Full LAT5150 integration
