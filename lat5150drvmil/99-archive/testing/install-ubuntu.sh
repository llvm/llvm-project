#!/bin/bash
# Distribution-specific installation script
# Generated for Ubuntu 24.04

set -e

DISTRO="Ubuntu"
VERSION="24.04"
KERNEL="6.14.0-29-generic"

echo "🔧 Installing DSMIL testing framework on $DISTRO $VERSION"

# Check root privileges for system operations
if [[ $EUID -eq 0 ]]; then
   echo "⚠️ Running as root. Some operations will be performed with reduced privileges."
fi

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update
sudo apt install -y \
    build-essential \
    libsmbios-dev \
    libsmbios-bin \
    linux-headers-6.14.0-29-generic \
    python3-pip \
    python3-psutil \
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
    echo "   Install: sudo apt install linux-headers-6.14.0-29-generic"
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
echo "2. make -f Makefile.ubuntu"
echo "3. sudo insmod dsmil-72dev.ko"
echo "4. python3 ../../testing/smbios_testbed_framework.py"
echo ""
