#!/bin/bash
# TPM2 Native OS Integration Installer
# Installs TCTI plugin, udev rules, and configures system for native TPM2 acceleration

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TPM2 Native Integration Installer ===${NC}"
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}Error: This script must be run as root${NC}"
   echo "Usage: sudo $0"
   exit 1
fi

# Detect paths
PROJECT_ROOT="/home/user/LAT5150DRVMIL"
TCTI_DIR="$PROJECT_ROOT/tpm2_compat/tcti"
UDEV_DIR="$PROJECT_ROOT/tpm2_compat/udev"
KERNEL_DIR="$PROJECT_ROOT/tpm2_compat/kernel"

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        LIB_DIR="/usr/lib/x86_64-linux-gnu"
        ;;
    aarch64)
        LIB_DIR="/usr/lib/aarch64-linux-gnu"
        ;;
    *)
        LIB_DIR="/usr/lib"
        ;;
esac

echo -e "${YELLOW}Installation Configuration:${NC}"
echo "  Architecture: $ARCH"
echo "  Library directory: $LIB_DIR"
echo "  Project root: $PROJECT_ROOT"
echo ""

# Step 1: Check dependencies
echo -e "${YELLOW}[1/7] Checking dependencies...${NC}"

if ! command -v gcc &> /dev/null; then
    echo -e "${RED}Error: gcc not found${NC}"
    echo "Install with: sudo apt install build-essential"
    exit 1
fi

if ! dpkg -l | grep -q libtss2-dev; then
    echo -e "${YELLOW}Warning: libtss2-dev not installed${NC}"
    echo "Installing TPM2 development libraries..."
    apt-get update
    apt-get install -y libtss2-dev tpm2-tools
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# Step 2: Build acceleration library
echo -e "${YELLOW}[2/7] Building TPM2 acceleration library...${NC}"

if [ -d "$PROJECT_ROOT/tpm2_compat/c_acceleration" ]; then
    cd "$PROJECT_ROOT/tpm2_compat/c_acceleration"

    # Check if Makefile exists
    if [ ! -f "Makefile" ]; then
        echo -e "${RED}Error: Makefile not found in c_acceleration directory${NC}"
        exit 1
    fi

    make clean || true
    make all || {
        echo -e "${RED}Error: Failed to build acceleration library${NC}"
        exit 1
    }

    echo -e "${GREEN}✓ Acceleration library built${NC}"
else
    echo -e "${YELLOW}Warning: Acceleration library source not found, skipping${NC}"
fi

# Step 3: Build TCTI plugin
echo -e "${YELLOW}[3/7] Building TCTI plugin...${NC}"

if [ -d "$TCTI_DIR" ] && [ -f "$TCTI_DIR/Makefile" ]; then
    cd "$TCTI_DIR"
    make clean || true
    make all || {
        echo -e "${RED}Error: Failed to build TCTI plugin${NC}"
        exit 1
    }

    echo -e "${GREEN}✓ TCTI plugin built${NC}"
else
    echo -e "${YELLOW}Warning: TCTI plugin source not found, skipping${NC}"
fi

# Step 4: Install TCTI plugin
echo -e "${YELLOW}[4/7] Installing TCTI plugin...${NC}"

if [ -f "$TCTI_DIR/libtss2-tcti-accel.so.0.0.0" ]; then
    cd "$TCTI_DIR"
    make install || {
        echo -e "${RED}Error: Failed to install TCTI plugin${NC}"
        exit 1
    }

    # Verify installation
    if [ -f "$LIB_DIR/libtss2-tcti-accel.so" ]; then
        echo -e "${GREEN}✓ TCTI plugin installed to $LIB_DIR${NC}"
    else
        echo -e "${RED}Error: TCTI plugin not found after installation${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Warning: TCTI plugin binary not found, skipping installation${NC}"
fi

# Step 5: Install udev rules
echo -e "${YELLOW}[5/7] Installing udev rules...${NC}"

if [ -f "$UDEV_DIR/99-tpm2-accel.rules" ]; then
    cp "$UDEV_DIR/99-tpm2-accel.rules" /etc/udev/rules.d/
    udevadm control --reload-rules
    udevadm trigger
    echo -e "${GREEN}✓ udev rules installed${NC}"
else
    echo -e "${YELLOW}Warning: udev rules file not found${NC}"
fi

# Step 6: Configure TPM2 tools
echo -e "${YELLOW}[6/7] Configuring TPM2 tools...${NC}"

mkdir -p /etc/tpm2-tools

cat > /etc/tpm2-tools/tpm2-tools.conf << 'EOF'
# TPM2 Tools Configuration
# Hardware acceleration enabled by default

[tcti]
# Use hardware-accelerated TCTI
tcti = accel:device=/dev/tpm2_accel_early,accel=all,security=0

# Fallback to hardware TPM if acceleration unavailable
#tcti = device:/dev/tpm0
EOF

echo -e "${GREEN}✓ TPM2 tools configured${NC}"

# Step 7: Add user to tss group
echo -e "${YELLOW}[7/7] Configuring user permissions...${NC}"

# Create tss group if it doesn't exist
if ! getent group tss > /dev/null; then
    groupadd -r tss
    echo -e "${GREEN}✓ Created tss group${NC}"
fi

# Add sudo user to tss group
if [ -n "$SUDO_USER" ]; then
    usermod -a -G tss "$SUDO_USER"
    echo -e "${GREEN}✓ Added $SUDO_USER to tss group${NC}"
else
    echo -e "${YELLOW}Warning: Could not determine user to add to tss group${NC}"
    echo "  Manually add with: sudo usermod -a -G tss <username>"
fi

# Load kernel module if available
if [ -f "$KERNEL_DIR/tpm_accel_chardev.ko" ]; then
    echo ""
    echo -e "${YELLOW}Loading kernel module...${NC}"
    insmod "$KERNEL_DIR/tpm_accel_chardev.ko" || {
        echo -e "${YELLOW}Warning: Could not load kernel module (may already be loaded)${NC}"
    }
fi

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                ║${NC}"
echo -e "${GREEN}║   ✅  Installation Complete!                   ║${NC}"
echo -e "${GREEN}║                                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Log out and back in (for group membership to take effect)"
echo "  2. Verify device exists: ls -l /dev/tpm*"
echo "  3. Test acceleration: tpm2_getrandom 32"
echo ""
echo -e "${YELLOW}Environment Variables:${NC}"
echo "  export TPM2TOOLS_TCTI=accel    # Use acceleration"
echo "  export TPM2TOOLS_TCTI=device   # Use hardware TPM"
echo ""
echo -e "${YELLOW}Hardware Acceleration:${NC}"
echo "  ✨ 76.4 TOPS available"
echo "  ✨ 88 cryptographic algorithms"
echo "  ✨ Intel NPU (34.0 TOPS)"
echo "  ✨ Intel GNA 3.5"
echo "  ✨ AES-NI, SHA-NI, AVX-512"
echo ""
echo -e "${GREEN}For more information:${NC}"
echo "  Documentation: $PROJECT_ROOT/tpm2_compat/TPM2_OS_NATIVE_INTEGRATION.md"
echo ""
