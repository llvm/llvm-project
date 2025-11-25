#!/bin/bash
#
# Intel NCS2 (Neural Compute Stick 2) Installation Script
# ========================================================
# Builds and installs the Movidius Myriad X VPU driver and Rust NCAPI
# for LAT5150DRVMIL AI Platform
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NCS2_DIR="$PROJECT_ROOT/04-hardware/ncs2-driver"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "========================================"
echo "  Intel NCS2 Driver Installation"
echo "  for LAT5150DRVMIL AI Platform"
echo "========================================"
echo ""

# Check if running as root (needed for kernel module)
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (for kernel module installation)"
    echo "Please run: sudo $0"
    exit 1
fi

# Detect NCS2 devices
log_info "Detecting Intel NCS2 devices..."
NCS2_COUNT=$(lsusb | grep -c "03e7:2485" || true)

if [ "$NCS2_COUNT" -eq 0 ]; then
    log_warning "No Intel NCS2 devices detected"
    log_info "Please connect Intel Neural Compute Stick 2 devices to USB ports"
    read -p "Continue installation anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installation cancelled"
        exit 0
    fi
else
    log_success "Found $NCS2_COUNT Intel NCS2 device(s)"
    lsusb | grep "03e7:2485"
fi

# Check kernel version
log_info "Checking kernel version..."
KERNEL_VERSION=$(uname -r | cut -d'.' -f1,2)
KERNEL_MAJOR=$(echo $KERNEL_VERSION | cut -d'.' -f1)
KERNEL_MINOR=$(echo $KERNEL_VERSION | cut -d'.' -f2)

if [ "$KERNEL_MAJOR" -lt 5 ] || ([ "$KERNEL_MAJOR" -eq 5 ] && [ "$KERNEL_MINOR" -lt 12 ]); then
    log_error "Kernel version $KERNEL_VERSION is too old. Requires >= 5.12 for io_uring support"
    exit 1
fi
log_success "Kernel version $KERNEL_VERSION OK (>= 5.12 required)"

# Check prerequisites
log_info "Checking prerequisites..."

# Check for kernel headers
if [ ! -d "/lib/modules/$(uname -r)/build" ]; then
    log_error "Kernel headers not found"
    log_info "Install with: apt install linux-headers-$(uname -r)"
    exit 1
fi
log_success "Kernel headers found"

# Check for build tools
if ! command -v make &> /dev/null; then
    log_error "Make not found. Install with: apt install build-essential"
    exit 1
fi
log_success "Build tools found"

# Check for Rust
if ! command -v cargo &> /dev/null; then
    log_warning "Rust/Cargo not found"
    log_info "Installing Rust..."

    # Install Rust for the actual user (not root)
    if [ -n "${SUDO_USER:-}" ]; then
        su - $SUDO_USER -c 'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y'
        export PATH="/home/$SUDO_USER/.cargo/bin:$PATH"
    else
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi

    if ! command -v cargo &> /dev/null; then
        log_error "Rust installation failed"
        exit 1
    fi
    log_success "Rust installed"
else
    RUST_VERSION=$(cargo --version | cut -d' ' -f2)
    log_success "Rust $RUST_VERSION found"
fi

# Navigate to NCS2 driver directory
if [ ! -d "$NCS2_DIR" ]; then
    log_error "NCS2 driver directory not found: $NCS2_DIR"
    log_info "Please ensure git submodule is initialized:"
    log_info "  git submodule update --init --recursive"
    exit 1
fi

cd "$NCS2_DIR"
log_info "Building in: $NCS2_DIR"

# Build kernel driver
log_info "Building kernel driver..."
make clean || true
make

if [ ! -f "movidius_x_vpu.ko" ]; then
    log_error "Kernel module build failed"
    exit 1
fi
log_success "Kernel driver built: movidius_x_vpu.ko"

# Check if module is already loaded
if lsmod | grep -q "movidius_x_vpu"; then
    log_info "Unloading existing module..."
    rmmod movidius_x_vpu || true
fi

# Load kernel module
log_info "Loading kernel module..."
insmod movidius_x_vpu.ko \
    batch_delay_ms=5 \
    batch_high_watermark=64 \
    submission_cpu_affinity=4

# Verify module loaded
if ! lsmod | grep -q "movidius_x_vpu"; then
    log_error "Failed to load kernel module"
    dmesg | tail -20
    exit 1
fi
log_success "Kernel module loaded"

# Check device nodes
DEVICE_COUNT=$(ls -1 /dev/movidius_x_vpu_* 2>/dev/null | wc -l || echo 0)
if [ "$DEVICE_COUNT" -eq 0 ]; then
    log_warning "No device nodes created"
    dmesg | tail -20
else
    log_success "Created $DEVICE_COUNT device node(s):"
    ls -l /dev/movidius_x_vpu_*
fi

# Build Rust components
log_info "Building Rust components..."
cd movidius-rs

# Build as the actual user (not root)
if [ -n "${SUDO_USER:-}" ]; then
    su - $SUDO_USER -c "cd $NCS2_DIR/movidius-rs && cargo build --release"
else
    cargo build --release
fi

if [ ! -f "target/release/movidius-bench" ]; then
    log_error "Rust components build failed"
    exit 1
fi
log_success "Rust components built"

# Install binaries
log_info "Installing binaries..."
INSTALL_DIR="/opt/ncs2"
mkdir -p "$INSTALL_DIR/bin"
mkdir -p "$INSTALL_DIR/lib"

# Copy binaries
cp target/release/movidius-bench "$INSTALL_DIR/bin/"
cp target/release/libmovidius_ncapi.so "$INSTALL_DIR/lib/" 2>/dev/null || true
chmod +x "$INSTALL_DIR/bin/movidius-bench"

log_success "Binaries installed to $INSTALL_DIR"

# Create systemd service for automatic module loading
log_info "Creating systemd service..."

cat > /etc/systemd/system/ncs2-driver.service << 'EOF'
[Unit]
Description=Intel NCS2 (Neural Compute Stick 2) Driver
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/insmod /opt/ncs2/movidius_x_vpu.ko batch_delay_ms=5 batch_high_watermark=64 submission_cpu_affinity=4
ExecStop=/sbin/rmmod movidius_x_vpu
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Copy kernel module to /opt/ncs2
cp "$NCS2_DIR/movidius_x_vpu.ko" "$INSTALL_DIR/"

systemctl daemon-reload
systemctl enable ncs2-driver.service
log_success "Systemd service created and enabled"

# Create environment setup script
cat > /etc/profile.d/ncs2.sh << EOF
# Intel NCS2 Environment
export NCS2_INSTALL_DIR="$INSTALL_DIR"
export PATH="\$NCS2_INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$NCS2_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
EOF

log_success "Environment script created: /etc/profile.d/ncs2.sh"

# Test devices
log_info "Testing NCS2 devices..."

if [ "$DEVICE_COUNT" -gt 0 ]; then
    echo ""
    echo "Device 0 Statistics:"
    echo "-------------------"

    # Temperature
    if [ -f "/sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/temperature" ]; then
        TEMP=$(cat /sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/temperature)
        echo "Temperature: ${TEMP}°C"
    fi

    # Firmware version
    if [ -f "/sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/firmware_version" ]; then
        FW_VER=$(cat /sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/firmware_version)
        echo "Firmware: $FW_VER"
    fi

    # Utilization
    if [ -f "/sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/compute_utilization" ]; then
        UTIL=$(cat /sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/compute_utilization)
        echo "Utilization: ${UTIL}%"
    fi
fi

# Installation summary
echo ""
echo "========================================"
echo "  Installation Summary"
echo "========================================"
echo ""
log_success "Kernel driver: movidius_x_vpu.ko (loaded)"
log_success "Rust NCAPI: $INSTALL_DIR/lib/"
log_success "Benchmark tool: $INSTALL_DIR/bin/movidius-bench"
log_success "Device nodes: $DEVICE_COUNT"
log_success "Systemd service: ncs2-driver.service (enabled)"
echo ""

# Usage instructions
echo "Next Steps:"
echo ""
echo "1. Reload environment variables:"
echo "   source /etc/profile.d/ncs2.sh"
echo ""
echo "2. Run benchmark tool:"
echo "   movidius-bench"
echo ""
echo "3. Monitor devices:"
echo "   cat /sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/temperature"
echo "   cat /sys/class/movidius_x_vpu/movidius_x_vpu_0/movidius/compute_utilization"
echo ""
echo "4. View module info:"
echo "   modinfo movidius_x_vpu"
echo ""
echo "5. Check dmesg for driver messages:"
echo "   dmesg | grep movidius"
echo ""

log_success "Intel NCS2 driver installation complete!"

# Return to project root
cd "$PROJECT_ROOT"
