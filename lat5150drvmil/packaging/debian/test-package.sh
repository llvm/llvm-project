#!/bin/bash
# Test script for tpm2-accel-early-dkms package
# Copyright (C) 2025 Military TPM2 Acceleration Project
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_FILE="${SCRIPT_DIR}/build/tpm2-accel-early-dkms_1.0.0-1_amd64.deb"

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_BLUE='\033[0;34m'
COLOR_RESET='\033[0m'

test_passed=0
test_failed=0

print_header() {
    echo -e "${COLOR_BLUE}=========================================${COLOR_RESET}"
    echo -e "${COLOR_BLUE}$1${COLOR_RESET}"
    echo -e "${COLOR_BLUE}=========================================${COLOR_RESET}"
}

print_test() {
    echo -e "${COLOR_YELLOW}[TEST]${COLOR_RESET} $1"
}

print_success() {
    echo -e "${COLOR_GREEN}[PASS]${COLOR_RESET} $1"
    ((test_passed++))
}

print_fail() {
    echo -e "${COLOR_RED}[FAIL]${COLOR_RESET} $1"
    ((test_failed++))
}

print_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $1"
}

# Test 1: Check if package exists
print_header "Test 1: Package File Verification"
print_test "Checking if package file exists..."
if [ -f "${PACKAGE_FILE}" ]; then
    print_success "Package file found: ${PACKAGE_FILE}"
    print_info "Size: $(du -h "${PACKAGE_FILE}" | cut -f1)"
else
    print_fail "Package file not found: ${PACKAGE_FILE}"
    echo "Run ./build-package.sh first"
    exit 1
fi

# Test 2: Verify package structure
print_header "Test 2: Package Structure Verification"
print_test "Verifying package metadata..."
if dpkg-deb -I "${PACKAGE_FILE}" >/dev/null 2>&1; then
    print_success "Package metadata is valid"
else
    print_fail "Invalid package metadata"
fi

# Test 3: Check package contents
print_header "Test 3: Package Contents Verification"
print_test "Checking for required files..."

REQUIRED_FILES=(
    "./DEBIAN/control"
    "./DEBIAN/postinst"
    "./DEBIAN/prerm"
    "./DEBIAN/postrm"
    "./DEBIAN/copyright"
    "./DEBIAN/changelog"
    "./DEBIAN/compat"
    "./DEBIAN/conffiles"
    "./lib/systemd/system/tpm2-acceleration-early.service"
    "./usr/src/tpm2-accel-early-1.0.0/dkms.conf"
    "./usr/src/tpm2-accel-early-1.0.0/tpm2_accel_early.c"
    "./usr/src/tpm2-accel-early-1.0.0/tpm2_accel_early.h"
    "./usr/src/tpm2-accel-early-1.0.0/Makefile"
)

PACKAGE_CONTENTS=$(dpkg-deb -c "${PACKAGE_FILE}")

for file in "${REQUIRED_FILES[@]}"; do
    if echo "${PACKAGE_CONTENTS}" | grep -q "${file}"; then
        print_success "Found: ${file}"
    else
        print_fail "Missing: ${file}"
    fi
done

# Test 4: Verify maintainer scripts
print_header "Test 4: Maintainer Scripts Verification"
print_test "Checking maintainer script permissions..."

TEMP_DIR=$(mktemp -d)
dpkg-deb -x "${PACKAGE_FILE}" "${TEMP_DIR}"
dpkg-deb -e "${PACKAGE_FILE}" "${TEMP_DIR}/DEBIAN"

for script in postinst prerm postrm; do
    if [ -x "${TEMP_DIR}/DEBIAN/${script}" ]; then
        print_success "${script} is executable"
    else
        print_fail "${script} is not executable"
    fi
done

# Test 5: Validate maintainer scripts syntax
print_test "Validating shell script syntax..."
for script in postinst prerm postrm; do
    if bash -n "${TEMP_DIR}/DEBIAN/${script}" 2>/dev/null; then
        print_success "${script} syntax is valid"
    else
        print_fail "${script} has syntax errors"
    fi
done

rm -rf "${TEMP_DIR}"

# Test 6: Check dependencies
print_header "Test 6: Dependency Verification"
print_test "Checking package dependencies..."

DEPS=$(dpkg-deb -f "${PACKAGE_FILE}" Depends)
print_info "Dependencies: ${DEPS}"

if echo "${DEPS}" | grep -q "dkms"; then
    print_success "DKMS dependency found"
else
    print_fail "DKMS dependency missing"
fi

# Test 7: Verify DKMS configuration
print_header "Test 7: DKMS Configuration Verification"
print_test "Extracting and checking DKMS config..."

TEMP_DIR=$(mktemp -d)
dpkg-deb -x "${PACKAGE_FILE}" "${TEMP_DIR}"

DKMS_CONF="${TEMP_DIR}/usr/src/tpm2-accel-early-1.0.0/dkms.conf"

if [ -f "${DKMS_CONF}" ]; then
    print_success "DKMS configuration file found"

    # Check required DKMS variables
    for var in PACKAGE_NAME PACKAGE_VERSION BUILT_MODULE_NAME DEST_MODULE_LOCATION; do
        if grep -q "^${var}=" "${DKMS_CONF}"; then
            print_success "DKMS variable ${var} is set"
        else
            print_fail "DKMS variable ${var} is missing"
        fi
    done
else
    print_fail "DKMS configuration file not found"
fi

rm -rf "${TEMP_DIR}"

# Test 8: Verify systemd service file
print_header "Test 8: Systemd Service Verification"
print_test "Extracting and checking systemd service..."

TEMP_DIR=$(mktemp -d)
dpkg-deb -x "${PACKAGE_FILE}" "${TEMP_DIR}"

SERVICE_FILE="${TEMP_DIR}/lib/systemd/system/tpm2-acceleration-early.service"

if [ -f "${SERVICE_FILE}" ]; then
    print_success "Systemd service file found"

    # Check required service sections
    for section in "[Unit]" "[Service]" "[Install]"; do
        if grep -q "^${section}" "${SERVICE_FILE}"; then
            print_success "Service section ${section} present"
        else
            print_fail "Service section ${section} missing"
        fi
    done

    # Validate systemd syntax (if systemd-analyze is available)
    if command -v systemd-analyze >/dev/null 2>&1; then
        if systemd-analyze verify "${SERVICE_FILE}" 2>/dev/null; then
            print_success "Systemd service syntax is valid"
        else
            print_fail "Systemd service has syntax errors"
        fi
    fi
else
    print_fail "Systemd service file not found"
fi

rm -rf "${TEMP_DIR}"

# Test 9: Verify source code files
print_header "Test 9: Source Code Verification"
print_test "Checking kernel module source files..."

TEMP_DIR=$(mktemp -d)
dpkg-deb -x "${PACKAGE_FILE}" "${TEMP_DIR}"

SOURCE_DIR="${TEMP_DIR}/usr/src/tpm2-accel-early-1.0.0"

# Check C source file
if [ -f "${SOURCE_DIR}/tpm2_accel_early.c" ]; then
    print_success "Kernel module C source found"
    LINES=$(wc -l < "${SOURCE_DIR}/tpm2_accel_early.c")
    print_info "Source lines: ${LINES}"
else
    print_fail "Kernel module C source missing"
fi

# Check header file
if [ -f "${SOURCE_DIR}/tpm2_accel_early.h" ]; then
    print_success "Kernel module header found"
else
    print_fail "Kernel module header missing"
fi

# Check Makefile
if [ -f "${SOURCE_DIR}/Makefile" ]; then
    print_success "Kernel module Makefile found"

    # Verify Makefile has required target
    if grep -q "^obj-m" "${SOURCE_DIR}/Makefile"; then
        print_success "Makefile has obj-m target"
    else
        print_fail "Makefile missing obj-m target"
    fi
else
    print_fail "Kernel module Makefile missing"
fi

rm -rf "${TEMP_DIR}"

# Test 10: Lintian checks (if available)
print_header "Test 10: Lintian Package Quality Checks"
if command -v lintian >/dev/null 2>&1; then
    print_test "Running lintian checks..."
    if lintian --no-tag-display-limit "${PACKAGE_FILE}" 2>&1 | tee /tmp/lintian.log; then
        print_info "Lintian output saved to /tmp/lintian.log"
        print_success "Lintian checks completed"
    else
        print_info "Lintian found issues (see /tmp/lintian.log)"
    fi
else
    print_info "Lintian not available (install with: apt install lintian)"
fi

# Final Summary
print_header "Test Summary"
TOTAL=$((test_passed + test_failed))
PERCENTAGE=$((test_passed * 100 / TOTAL))

echo ""
echo -e "Total Tests:  ${TOTAL}"
echo -e "${COLOR_GREEN}Passed:       ${test_passed}${COLOR_RESET}"
echo -e "${COLOR_RED}Failed:       ${test_failed}${COLOR_RESET}"
echo -e "Success Rate: ${PERCENTAGE}%"
echo ""

if [ ${test_failed} -eq 0 ]; then
    print_header "All Tests Passed!"
    echo -e "${COLOR_GREEN}Package is ready for deployment${COLOR_RESET}"
    echo ""
    echo "Installation command:"
    echo "  sudo dpkg -i ${PACKAGE_FILE}"
    exit 0
else
    print_header "Some Tests Failed"
    echo -e "${COLOR_RED}Please review failures before deployment${COLOR_RESET}"
    exit 1
fi
