#!/bin/bash
# Build script for tpm2-accel-early-dkms Debian package
# Copyright (C) 2025 Military TPM2 Acceleration Project
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY

set -e

PACKAGE_NAME="tpm2-accel-early"
PACKAGE_VERSION="1.0.0"
PACKAGE_REVISION="1"
PACKAGE_FULL="${PACKAGE_NAME}-dkms_${PACKAGE_VERSION}-${PACKAGE_REVISION}_amd64"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
PACKAGE_DIR="${BUILD_DIR}/${PACKAGE_NAME}-dkms_${PACKAGE_VERSION}-${PACKAGE_REVISION}_amd64"
SOURCE_DIR="${SCRIPT_DIR}/../../tpm2_compat/c_acceleration/kernel_module"
DKMS_CONF="${SCRIPT_DIR}/../dkms/tpm2-accel-early.dkms.conf"

echo "========================================="
echo "Building ${PACKAGE_FULL}.deb"
echo "========================================="

# Clean previous builds
if [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning previous build..."
    rm -rf "${BUILD_DIR}"
fi

# Create package directory structure
echo "Creating package structure..."
mkdir -p "${PACKAGE_DIR}"

# Copy DEBIAN control files
echo "Copying DEBIAN control files..."
cp -a "${SCRIPT_DIR}/DEBIAN" "${PACKAGE_DIR}/"

# Create usr/src directory for DKMS
echo "Creating DKMS source tree..."
mkdir -p "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}"

# Copy kernel module source
echo "Copying kernel module source..."
cp "${SOURCE_DIR}/tpm2_accel_early.c" "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}/"
cp "${SOURCE_DIR}/tpm2_accel_early.h" "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}/"
cp "${SOURCE_DIR}/Makefile" "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}/"

# Copy DKMS configuration
echo "Copying DKMS configuration..."
cp "${DKMS_CONF}" "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}/dkms.conf"

# Copy systemd service
echo "Copying systemd service..."
mkdir -p "${PACKAGE_DIR}/lib/systemd/system"
cp "${SCRIPT_DIR}/lib/systemd/system/tpm2-acceleration-early.service" \
   "${PACKAGE_DIR}/lib/systemd/system/"

# Create documentation directory
echo "Creating documentation..."
mkdir -p "${PACKAGE_DIR}/usr/share/doc/${PACKAGE_NAME}-dkms"

cat > "${PACKAGE_DIR}/usr/share/doc/${PACKAGE_NAME}-dkms/README.Debian" <<'EOF'
tpm2-accel-early-dkms for Debian
=================================

This package provides TPM2 hardware acceleration for early boot using DKMS.

Installation
------------

After installation, the module is automatically built and loaded. You can
verify the installation with:

    lsmod | grep tpm2_accel_early
    dmesg | grep tpm2_accel_early

The character device should appear at /dev/tpm2_accel_early

Configuration
-------------

The module accepts the following parameters:

  - security_level: 0=UNCLASSIFIED (default), 1=CONFIDENTIAL, 2=SECRET, 3=TOP_SECRET
  - early_init: 1=enable early boot initialization (default), 0=disable
  - debug_mode: 0=disabled (default), 1=enabled

To change these parameters, edit /etc/modprobe.d/tpm2-acceleration.conf
and run:

    update-initramfs -u
    reboot

Hardware Requirements
---------------------

  - Intel Core Ultra 7 165H (Meteor Lake) or compatible
  - TPM 2.0 hardware module
  - Dell Latitude 5450 MIL-SPEC or compatible system

Security Considerations
-----------------------

This module integrates with Dell SMBIOS military tokens (0x049e-0x04a3) for
hardware-level access control. Ensure your system has appropriate Dell SMBIOS
support before deploying at elevated security levels.

Troubleshooting
---------------

If the module fails to load:

  1. Check kernel logs: journalctl -k | grep tpm2_accel_early
  2. Verify TPM is available: ls -l /dev/tpm*
  3. Check DKMS status: dkms status tpm2-accel-early
  4. Rebuild module: dkms build -m tpm2-accel-early -v 1.0.0

For more information, see the upstream documentation.

 -- Military TPM2 Acceleration Project <noreply@example.com>
EOF

# Copy changelog and compress
cp "${SCRIPT_DIR}/DEBIAN/changelog" "${PACKAGE_DIR}/usr/share/doc/${PACKAGE_NAME}-dkms/changelog.Debian"
gzip -9 "${PACKAGE_DIR}/usr/share/doc/${PACKAGE_NAME}-dkms/changelog.Debian"

# Copy copyright
cp "${SCRIPT_DIR}/DEBIAN/copyright" "${PACKAGE_DIR}/usr/share/doc/${PACKAGE_NAME}-dkms/"

# Set proper permissions
echo "Setting permissions..."
find "${PACKAGE_DIR}" -type d -exec chmod 0755 {} \;
find "${PACKAGE_DIR}/usr/src/${PACKAGE_NAME}-${PACKAGE_VERSION}" -type f -exec chmod 0644 {} \;
chmod 0755 "${PACKAGE_DIR}/DEBIAN"/{postinst,prerm,postrm}
chmod 0644 "${PACKAGE_DIR}/DEBIAN"/{control,copyright,changelog,compat,conffiles,triggers}
chmod 0644 "${PACKAGE_DIR}/lib/systemd/system/tpm2-acceleration-early.service"

# Build the package
echo "Building Debian package..."
cd "${BUILD_DIR}"
dpkg-deb --build --root-owner-group "${PACKAGE_FULL}"

# Verify package
echo ""
echo "========================================="
echo "Package built successfully!"
echo "========================================="
echo "Package: ${BUILD_DIR}/${PACKAGE_FULL}.deb"
echo ""
echo "Package information:"
dpkg-deb -I "${PACKAGE_FULL}.deb"
echo ""
echo "Package contents:"
dpkg-deb -c "${PACKAGE_FULL}.deb"
echo ""
echo "========================================="
echo "Installation command:"
echo "  sudo dpkg -i ${PACKAGE_FULL}.deb"
echo "  sudo apt-get install -f  # if dependencies missing"
echo "========================================="
echo ""
