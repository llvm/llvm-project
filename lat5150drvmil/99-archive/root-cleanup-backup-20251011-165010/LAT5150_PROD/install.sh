#!/bin/bash
# LAT5150 DSMIL Production Installer

set -e

echo "Installing LAT5150 DSMIL Production System..."

# Check for root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

# Install kernel modules
echo "Installing kernel modules..."
cp bin/*.ko /lib/modules/$(uname -r)/extra/
depmod -a

# Install libraries
echo "Installing libraries..."
cp lib/* /usr/local/lib/
ldconfig

# Install binaries
echo "Installing binaries..."
cp bin/* /usr/local/bin/ 2>/dev/null || true

# Install configuration
echo "Installing configuration..."
mkdir -p /etc/lat5150
cp config/* /etc/lat5150/

# Load TPM module if available
if [ -f /dev/tpm0 ]; then
    echo "TPM device detected, enabling TPM integration..."
    echo "tpm_tis" >> /etc/modules-load.d/tpm.conf
fi

# Load DSMIL module
echo "Loading DSMIL module..."
modprobe dsmil-72dev || echo "DSMIL module load failed - check dmesg"

echo "Installation complete!"
echo "Check system logs and run validation tests."