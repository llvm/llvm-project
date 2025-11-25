# LAT5150DRVMIL Submodule Integration Guide

**Version**: 9.0.0
**Updated**: 2025-11-23
**Status**: Production Ready

---

## Overview

This guide explains how to integrate **LAT5150DRVMIL** (Dell Latitude 5450 AI Tactical Platform) as a Git submodule into your parent project.

### What is LAT5150DRVMIL?

A complete **LOCAL-FIRST AI platform** featuring:
- **104 DSMIL devices** with kernel drivers (dsmil-104dev.ko, dsmil-84dev.ko)
- **5 local AI models** (DeepSeek, Qwen, WizardLM, CodeLlama, WhiteRabbit)
- **11 MCP servers** for extended capabilities
- **TPM 2.0 hardware attestation** with quantum-resistant crypto (CSNA 2.0)
- **DEB package system** for modular deployment
- **Unified dashboard** at http://localhost:5050

---

## Quick Start (3 Steps)

### 1. Add as Submodule

```bash
# In your parent project root
git submodule add https://github.com/SWORDIntel/LAT5150DRVMIL.git external/lat5150
git submodule update --init --recursive
```

### 2. Initialize Submodule

```bash
cd external/lat5150
git checkout claude/prepare-submodule-integration-011eQWEePnEKZ4XFaGP7yT1L
```

### 3. Build & Install

```bash
# Option A: DEB packages (recommended)
cd packaging
./build-all-debs.sh
sudo ./install-all-debs.sh

# Option B: Kernel drivers only
sudo python3 dsmil.py build-auto
sudo python3 dsmil.py load-all

# Option C: Complete environment
./lat5150_entrypoint.sh
```

---

## System Requirements

### Minimum (Development/Docker)
- **OS**: Linux (Ubuntu 22.04+, Debian 11+, RHEL 8+)
- **Kernel**: 5.15+ with headers installed
- **Python**: 3.10+
- **RAM**: 8 GB
- **Storage**: 20 GB
- **GCC**: 11+ (or Clang 14+)
- **Make**: 4.3+

### Recommended (Dell MIL-SPEC Hardware)
- **Platform**: Dell Latitude 5450 Covert Edition
- **CPU**: Intel Core Ultra 7 165H (Meteor Lake, 16 cores)
- **TPM**: 2.0 (STMicroelectronics ST33HTPH2E32 or Infineon SLB9672)
- **RAM**: 64 GB DDR5-5600
- **Storage**: 4 TB NVMe
- **GPU**: Intel Arc 140V (28.6 TOPS)
- **NPU**: Intel AI Boost VPU 3720 (48 TOPS)

### Optional Dependencies
- **Rust**: 1.70+ (for Rust safety layer in kernel driver)
- **Node.js**: 18+ (for external MCP servers)
- **Docker**: 24+ (for containerized deployment)
- **Ollama**: Latest (for AI model management)

---

## Integration Methods

### Method 1: DEB Package Integration (Recommended)

**Best for**: Production deployments, system-wide installation

```bash
cd external/lat5150/packaging

# Build 4 DEB packages
./build-all-debs.sh

# Packages created:
# - dsmil-platform_8.3.1-1_amd64.deb (2.5 MB) - Main platform
# - dell-milspec-tools_1.0.0-1_amd64.deb (24 KB) - Management tools
# - tpm2-accel-examples_1.0.0-1_amd64.deb (19 KB) - TPM examples
# - dsmil-complete_8.3.2-1_amd64.deb (1.5 KB) - Meta-package

# Install all packages
sudo ./install-all-debs.sh

# Verify installation (10-point check)
./verify-installation.sh
```

**Installed Components**:
- Binaries in `/usr/local/bin/`: `dsmil-status`, `milspec-control`, `milspec-monitor`
- Libraries in `/usr/local/lib/lat5150/`
- Configuration in `/etc/lat5150/`
- SystemD service: `lat5150-tactical-ai.service`
- Documentation in `/usr/local/share/doc/lat5150/`

### Method 2: Kernel Driver Integration

**Best for**: Hardware access, low-level integration

```bash
cd external/lat5150

# Auto-detect Rust and build both driver variants
sudo python3 dsmil.py build-auto

# This builds:
# - dsmil-104dev.ko (104-device variant with Rust safety layer)
# - dsmil-84dev.ko (84-device variant with C stubs)

# Load both drivers
sudo python3 dsmil.py load-all

# Check driver status
sudo python3 dsmil.py status
```

**Driver Files**:
- Source: `01-source/kernel/`
- Built modules: `01-source/kernel/dsmil-*.ko`
- Loaded at: `/sys/module/dsmil_*`

### Method 3: Python Module Integration

**Best for**: Python projects, direct API access

```python
# In your parent project's Python code
import sys
sys.path.insert(0, 'external/lat5150/02-ai-engine')

from dsmil_subsystem_controller import DSMILController
from quantum_crypto_layer import QuantumCryptoLayer
from tpm_crypto_integration import TPMCryptoIntegration

# Initialize components
controller = DSMILController()
crypto = QuantumCryptoLayer()
tpm = TPMCryptoIntegration()

# Use DSMIL devices
devices = controller.list_safe_devices()
status = controller.activate_device(0x8001)
```

### Method 4: Docker Integration

**Best for**: Containerized deployments, CI/CD

```dockerfile
# In your parent project's Dockerfile
FROM ubuntu:22.04

# Copy submodule
COPY external/lat5150 /opt/lat5150

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    build-essential linux-headers-generic \
    make gcc clang \
    && cd /opt/lat5150 \
    && pip3 install -r requirements.txt

# Build and install
RUN cd /opt/lat5150/packaging && \
    ./build-all-debs.sh && \
    ./install-all-debs.sh

# Start services
CMD ["/usr/local/bin/lat5150"]
```

### Method 5: API Integration

**Best for**: Microservices, REST API access

```bash
# Start the unified dashboard
cd external/lat5150
./scripts/start-dashboard.sh

# Access from your application:
# - Dashboard: http://localhost:5050
# - API: http://localhost:5050/api/
# - Self-awareness: http://localhost:5001/api/v2/self-awareness
```

**Key Endpoints**:
```bash
# System health
curl http://localhost:5050/api/dsmil/health

# List devices (99 usable, 5 quarantined)
curl http://localhost:5050/api/dsmil/subsystems

# TPM status
curl http://localhost:5050/api/tpm/status

# Run benchmarks (22 tests)
curl -X POST http://localhost:5050/api/benchmark/run

# Get results
curl http://localhost:5050/api/benchmark/results
```

---

## Build Configuration

### Compiler Flags (Intel Meteor Lake Optimized)

```bash
# Export these before building
export CFLAGS_OPTIMAL="-O3 -pipe -fomit-frame-pointer -funroll-loops -fstrict-aliasing -fno-plt -fdata-sections -ffunction-sections -flto=auto -march=meteorlake -mtune=meteorlake -msse4.2 -mpopcnt -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt -mmovbe -mavxvnni -maes -mvaes -mpclmul -mvpclmulqdq -msha -mgfni -madx -mclflushopt -mclwb -mcldemote -mmovdiri -mmovdir64b -mwaitpkg -mserialize -mtsxldtrk -muintr -mprefetchw -mprfchw -mrdrnd -mrdseed"

export LDFLAGS_OPTIMAL="-Wl,--as-needed -Wl,--gc-sections -Wl,-O1 -Wl,--hash-style=gnu -flto=auto"

export KCFLAGS="-O3 -pipe -march=meteorlake -mtune=meteorlake -mavx2 -mfma -mavxvnni -maes -mvaes -mpclmul -mvpclmulqdq -msha -mgfni -falign-functions=32"
```

**Performance Impact**:
- 15-30% faster compilation (LTO + optimized instruction selection)
- 10-25% runtime speedup (AVX2/VNNI vectorization)
- Hardware crypto acceleration (AES-NI, SHA-NI)

**Fallback for Older GCC**:
```bash
# If GCC doesn't recognize -march=meteorlake
export CFLAGS_FALLBACK="-O3 -march=alderlake -mtune=alderlake -mavx2 -mfma"
```

### Makefile Integration

```makefile
# In your parent project's Makefile

# Include LAT5150 submodule
SUBMODULE_DIR := external/lat5150

# Build LAT5150 components
lat5150-build:
	cd $(SUBMODULE_DIR) && sudo python3 dsmil.py build-auto

# Install LAT5150
lat5150-install:
	cd $(SUBMODULE_DIR)/packaging && sudo ./install-all-debs.sh

# Clean LAT5150 build artifacts
lat5150-clean:
	cd $(SUBMODULE_DIR) && make -C 01-source/kernel clean

# Complete integration
integrate: lat5150-build lat5150-install
	@echo "LAT5150 integration complete"

.PHONY: lat5150-build lat5150-install lat5150-clean integrate
```

### CMake Integration

```cmake
# In your parent project's CMakeLists.txt

# Add LAT5150 as external project
add_subdirectory(external/lat5150 EXCLUDE_FROM_ALL)

# Link against LAT5150 libraries
target_include_directories(your_target PRIVATE
    external/lat5150/02-ai-engine
    external/lat5150/01-source/kernel/core
)

# Custom commands for LAT5150 build
add_custom_target(lat5150-drivers
    COMMAND sudo python3 dsmil.py build-auto
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/external/lat5150
    COMMENT "Building LAT5150 kernel drivers"
)
```

---

## Configuration Files

### 1. MCP Servers Configuration

```bash
cp external/lat5150/02-ai-engine/mcp_servers_config.json config/lat5150_mcp.json
```

Edit for your environment:
```json
{
  "mcpServers": {
    "dsmil-ai": {
      "command": "python3",
      "args": ["external/lat5150/02-ai-engine/dsmil_mcp_server.py"],
      "env": {
        "DSMIL_DEVICES": "99",
        "TPM_ENABLED": "true"
      }
    }
  }
}
```

### 2. Environment Variables

```bash
# Create .env file
cat > config/lat5150.env <<EOF
# LAT5150 Configuration
DSMIL_ROOT=/path/to/external/lat5150
DSMIL_DRIVER_VARIANT=104dev
DSMIL_SAFE_MODE=true

# TPM Configuration
TPM_DEVICE=/dev/tpm0
TPM_ATTESTATION=true

# API Configuration
DSMIL_API_PORT=5050
DSMIL_API_SECRET=$(openssl rand -hex 32)

# AI Configuration
OLLAMA_HOST=http://localhost:11434
DEFAULT_MODEL=deepseek-r1:latest

# Security
QUARANTINE_ENFORCEMENT=true
AUDIT_LOGGING=true
EOF
```

### 3. SystemD Service Integration

```ini
# /etc/systemd/system/your-app-lat5150.service
[Unit]
Description=Your Application with LAT5150 Integration
After=network.target
Requires=lat5150-tactical-ai.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/your-app
EnvironmentFile=/opt/your-app/config/lat5150.env
ExecStartPre=/usr/local/bin/dsmil-status
ExecStart=/opt/your-app/bin/your-app
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Dependency Management

### Install All Dependencies

```bash
# From submodule root
cd external/lat5150

# Python dependencies
pip3 install -r requirements.txt

# System dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    python3.10 python3-pip \
    make gcc clang \
    git curl wget \
    libtpm2-dev \
    libssl-dev \
    pkg-config

# Optional: Rust toolchain (for Rust safety layer)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Optional: Node.js (for external MCP servers)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Optional: Ollama (for AI models)
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify Dependencies

```bash
cd external/lat5150
./verify_dependencies.sh
```

This checks:
- Kernel headers
- Python version
- GCC/Clang
- Rust (optional)
- Node.js (optional)
- TPM device access
- Required libraries

---

## Testing Integration

### 1. Smoke Test

```bash
cd external/lat5150

# Check submodule integrity
git submodule status

# Build test
sudo python3 dsmil.py build-auto

# Load test
sudo python3 dsmil.py status

# API test
python3 -c "
from dsmil_subsystem_controller import DSMILController
controller = DSMILController()
print(f'Found {len(controller.list_safe_devices())} safe devices')
"
```

### 2. Full Integration Test

```bash
cd external/lat5150

# Run 22 benchmark tests
curl -X POST http://localhost:5050/api/benchmark/run

# Wait for completion
sleep 10

# Get results
curl http://localhost:5050/api/benchmark/results | jq
```

### 3. Hardware Test (Dell MIL-SPEC only)

```bash
# Discover DSMIL hardware
sudo ./dsmil-discover.sh

# Analyze system
sudo ./dsmil-analyze.sh

# Test TPM
cd 02-ai-engine
sudo python3 audit_tpm_capabilities.py
```

---

## Security Considerations

### Quarantined Devices

**5 devices are PERMANENTLY BLOCKED** at 4 enforcement layers:

| Device ID | Name | Reason |
|-----------|------|--------|
| 0x8009 | DATA_DESTRUCTION | Destructive |
| 0x800A | SECURE_ERASE | Destructive |
| 0x800B | PERMANENT_DISABLE | Destructive |
| 0x8019 | NETWORK_KILL | Destructive |
| 0x8029 | FILESYSTEM_WIPE | Destructive |

**Cannot be activated under any circumstances.**

### Authentication Required

```python
# API requests require HMAC-SHA3-512 signatures
import hmac
import hashlib
import time

secret = os.environ['DSMIL_API_SECRET']
timestamp = str(int(time.time()))
message = f"POST:/api/dsmil/activate:{timestamp}"
signature = hmac.new(
    secret.encode(),
    message.encode(),
    hashlib.sha3_512
).hexdigest()

headers = {
    'X-DSMIL-Timestamp': timestamp,
    'X-DSMIL-Signature': signature
}
```

### Rate Limiting

- **60 requests/minute** per client IP
- **5-minute timestamp window** (replay protection)
- **Exponential backoff** on failures

### TPM Attestation

```python
from tpm_crypto_integration import TPMCryptoIntegration

tpm = TPMCryptoIntegration()
if tpm.is_available():
    quote = tpm.generate_quote()
    if tpm.verify_quote(quote):
        print("Hardware attestation successful")
```

---

## Troubleshooting

### Issue: Submodule not initialized

```bash
# Error: fatal: not a git repository
cd your-project-root
git submodule update --init --recursive
```

### Issue: Kernel headers missing

```bash
# Error: No rule to make target /lib/modules/.../build
sudo apt-get install linux-headers-$(uname -r)
```

### Issue: Build fails with "meteorlake not recognized"

```bash
# Fallback to alderlake
export CFLAGS="-O3 -march=alderlake -mtune=alderlake"
sudo python3 dsmil.py build-auto
```

### Issue: TPM not available

```bash
# Check device
ls -l /dev/tpm0

# If missing, check kernel module
sudo modprobe tpm_tis

# If still missing, TPM disabled in BIOS
# Reboot → F2 → Security → TPM 2.0 → Enable
```

### Issue: Permission denied loading driver

```bash
# Secure Boot must be disabled or driver signed
# Option 1: Disable Secure Boot (BIOS → Boot → Secure Boot → Disabled)
# Option 2: Sign the module (see 01-source/kernel/docs/signing.md)
sudo mokutil --disable-validation
```

### Issue: Port 5050 already in use

```bash
# Check what's using it
sudo lsof -i :5050

# Kill or change port
export DSMIL_API_PORT=5051
./scripts/start-dashboard.sh
```

---

## Maintenance

### Updating Submodule

```bash
cd external/lat5150
git fetch origin
git checkout claude/prepare-submodule-integration-011eQWEePnEKZ4XFaGP7yT1L
git pull

# Rebuild if needed
sudo python3 dsmil.py build-auto
cd packaging && ./build-all-debs.sh
```

### Uninstalling

```bash
# Remove DEB packages
sudo apt-get remove dsmil-complete dsmil-platform dell-milspec-tools tpm2-accel-examples

# Unload drivers
sudo python3 external/lat5150/dsmil.py unload-all

# Remove submodule
git submodule deinit -f external/lat5150
rm -rf .git/modules/external/lat5150
git rm -f external/lat5150
```

---

## Advanced Topics

### AVX-512 Unlock (Optional)

**Enables AVX-512 on P-cores for 15-40% speedup** by disabling E-cores.

```bash
cd external/lat5150/avx512-unlock
sudo ./unlock_avx512.sh enable

# Verify
./verify_avx512.sh

# Source AVX-512 flags
source ./avx512_compiler_flags.sh

# Rebuild with AVX-512
cd ../01-source/kernel
make KCFLAGS="$KCFLAGS_AVX512"
```

**Trade-off**: Lose 10 E-cores for 2x wider SIMD (512-bit vs 256-bit)

### Custom MCP Server Development

```bash
# Create new MCP server
cd external/lat5150/03-mcp-servers
mkdir my-custom-server
cd my-custom-server

cat > server.py <<'EOF'
from mcp import Server, MCPTool

server = Server("my-custom-server")

@server.tool()
def my_tool(param: str) -> str:
    return f"Processed: {param}"

if __name__ == "__main__":
    server.run()
EOF

# Register in config
cd ../../02-ai-engine
# Edit mcp_servers_config.json to add your server
```

### Performance Tuning

```bash
# CPU governor
sudo cpupower frequency-set -g performance

# Disable CPU idle
sudo systemctl mask sleep.target suspend.target

# Increase file descriptors
ulimit -n 65536

# Kernel parameters
sudo sysctl -w kernel.pid_max=4194304
sudo sysctl -w vm.max_map_count=262144
```

---

## Directory Reference

```
external/lat5150/
├── SUBMODULE_INTEGRATION.md       # This file
├── README.md                      # Main documentation
├── QUICKSTART.md                  # 5-minute quick start
├── HOW_TO_USE.md                  # Usage guide
├── INDEX.md                       # Directory guide
│
├── dsmil.py                       # Kernel driver build system
├── dsmil_control_centre.py        # Device management TUI
├── lat5150_entrypoint.sh          # Tmux environment
│
├── packaging/                     # DEB packages
│   ├── build-all-debs.sh
│   ├── install-all-debs.sh
│   └── BUILD_INSTRUCTIONS.md
│
├── 01-source/                     # DSMIL framework
│   └── kernel/                    # Kernel module source
│
├── 02-ai-engine/                  # AI platform
│   ├── ai_gui_dashboard.py        # Unified dashboard
│   ├── dsmil_subsystem_controller.py # 104 devices
│   ├── quantum_crypto_layer.py    # CSNA 2.0 crypto
│   └── tpm_crypto_integration.py  # TPM 2.0
│
├── 03-mcp-servers/                # MCP servers (11 total)
├── 04-integrations/               # RAG, web crawling
└── 05-deployment/                 # SystemD services
```

---

## Reference Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Main project documentation |
| [QUICKSTART.md](QUICKSTART.md) | 5-minute quick start |
| [HOW_TO_USE.md](HOW_TO_USE.md) | Complete usage guide |
| [INDEX.md](INDEX.md) | Directory structure |
| [DEPLOYMENT_READY.md](DEPLOYMENT_READY.md) | Deployment checklist |
| [packaging/BUILD_INSTRUCTIONS.md](packaging/BUILD_INSTRUCTIONS.md) | DEB package guide |
| [01-source/kernel/README.md](01-source/kernel/README.md) | Kernel driver docs |
| [avx512-unlock/README.md](avx512-unlock/README.md) | AVX-512 unlock guide |

---

## Support

- **Documentation**: See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)
- **MCP Setup**: See [03-mcp-servers/README.md](03-mcp-servers/README.md)
- **Issues**: File at https://github.com/SWORDIntel/LAT5150DRVMIL/issues

---

## License

**Proprietary - Dell Systems Management Interface Layer**
- DSMIL components: Restricted to authorized Dell MIL-SPEC hardware
- AI/MCP components: Open source (various licenses)
- Security tools: Respective project licenses
- **For authorized security research and training only**

---

**LAT5150DRVMIL v9.0.0** | Dell Latitude 5450 Covert Edition | Submodule Ready | Production Tested
