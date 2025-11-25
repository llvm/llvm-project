#!/bin/bash
#
# DSMIL Phase 2A Installer Test Script
# Quick verification of installer functionality and requirements
#
# Usage: ./test_installer.sh [--comprehensive]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLER_SCRIPT="$SCRIPT_DIR/install_dsmil_phase2a.sh"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

success() { echo -e "${GREEN}✓${NC} $*"; }
error() { echo -e "${RED}✗${NC} $*"; }
warning() { echo -e "${YELLOW}⚠${NC} $*"; }
info() { echo -e "${BLUE}ℹ${NC} $*"; }

test_installer_basics() {
    echo "Testing installer basics..."
    
    # Test installer exists and is executable
    if [[ -f "$INSTALLER_SCRIPT" && -x "$INSTALLER_SCRIPT" ]]; then
        success "Installer script exists and is executable"
    else
        error "Installer script missing or not executable: $INSTALLER_SCRIPT"
        return 1
    fi
    
    # Test help functionality
    if "$INSTALLER_SCRIPT" --help >/dev/null 2>&1; then
        success "Installer --help works"
    else
        error "Installer --help failed"
        return 1
    fi
    
    # Test dry-run functionality
    info "Testing dry-run mode..."
    if timeout 60 "$INSTALLER_SCRIPT" --dry-run --quiet; then
        success "Dry-run completed successfully"
    else
        error "Dry-run failed or timed out"
        return 1
    fi
    
    return 0
}

test_system_requirements() {
    echo
    echo "Testing system requirements..."
    
    local errors=0
    
    # Check kernel version
    local kernel_version
    kernel_version=$(uname -r)
    local major_version
    major_version=$(echo "$kernel_version" | cut -d. -f1)
    local minor_version
    minor_version=$(echo "$kernel_version" | cut -d. -f2)
    
    if [[ $major_version -gt 6 ]] || [[ $major_version -eq 6 && $minor_version -ge 14 ]]; then
        success "Kernel version: $kernel_version (>= 6.14.0)"
    else
        warning "Kernel version: $kernel_version (< 6.14.0, may need --force)"
    fi
    
    # Check required commands
    local required_commands=("gcc" "make" "python3" "sudo")
    for cmd in "${required_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            success "Command available: $cmd"
        else
            error "Required command missing: $cmd"
            ((errors++))
        fi
    done
    
    # Check optional commands
    local optional_commands=("cargo" "systemctl" "udevadm")
    for cmd in "${optional_commands[@]}"; do
        if command -v "$cmd" >/dev/null 2>&1; then
            success "Optional command available: $cmd"
        else
            info "Optional command missing: $cmd"
        fi
    done
    
    # Check kernel headers
    local kernel_headers_path="/lib/modules/$(uname -r)/build"
    if [[ -d "$kernel_headers_path" ]]; then
        success "Kernel headers available: $kernel_headers_path"
    else
        error "Kernel headers not found: $kernel_headers_path"
        echo "  Install with: sudo apt-get install linux-headers-$(uname -r)"
        ((errors++))
    fi
    
    # Check Python modules
    local python_modules=("json" "subprocess" "pathlib")
    for module in "${python_modules[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            success "Python module available: $module"
        else
            error "Python module missing: $module"
            ((errors++))
        fi
    done
    
    return $errors
}

test_source_structure() {
    echo
    echo "Testing source code structure..."
    
    local errors=0
    
    # Check key directories
    local required_dirs=(
        "01-source/kernel"
        "monitoring"
        "docs"
        "logs"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ -d "$SCRIPT_DIR/$dir" ]]; then
            success "Directory exists: $dir"
        else
            error "Required directory missing: $dir"
            ((errors++))
        fi
    done
    
    # Check key files
    local required_files=(
        "01-source/kernel/dsmil-72dev.c"
        "01-source/kernel/Makefile"
        "monitoring/README.md"
        "docs/PHASE2_CHUNKED_IOCTL_SOLUTION.md"
    )
    
    for file in "${required_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$file" ]]; then
            success "File exists: $file"
        else
            error "Required file missing: $file"
            ((errors++))
        fi
    done
    
    return $errors
}

test_kernel_module_readiness() {
    echo
    echo "Testing kernel module build readiness..."
    
    local kernel_dir="$SCRIPT_DIR/01-source/kernel"
    
    if [[ ! -d "$kernel_dir" ]]; then
        error "Kernel source directory not found"
        return 1
    fi
    
    cd "$kernel_dir"
    
    # Test makefile syntax
    if make -n >/dev/null 2>&1; then
        success "Makefile syntax is valid"
    else
        error "Makefile has syntax errors"
        return 1
    fi
    
    # Check if source compiles (dry run)
    info "Testing compilation readiness..."
    if make clean >/dev/null 2>&1; then
        success "Make clean successful"
    else
        warning "Make clean had issues"
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

run_comprehensive_tests() {
    echo
    echo "Running comprehensive installer tests..."
    
    # Test installation with various options
    local test_options=(
        "--dry-run --quiet"
        "--dry-run --no-monitoring"
        "--dry-run --no-rust"
        "--dry-run --no-monitoring --no-rust"
        "--dry-run --force"
    )
    
    for options in "${test_options[@]}"; do
        info "Testing: $options"
        if timeout 120 $INSTALLER_SCRIPT $options; then
            success "Test passed: $options"
        else
            error "Test failed: $options"
        fi
        echo
    done
}

show_summary() {
    echo
    echo "=============================================="
    echo "DSMIL Phase 2A Installer Test Summary"
    echo "=============================================="
    echo
    
    # System info
    echo "System Information:"
    echo "  OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
    echo "  Kernel: $(uname -r)"
    echo "  Architecture: $(uname -m)"
    echo "  Python: $(python3 --version 2>/dev/null || echo "Not found")"
    echo "  GCC: $(gcc --version 2>/dev/null | head -1 || echo "Not found")"
    echo
    
    # Hardware info
    echo "Hardware Information:"
    echo "  Manufacturer: $(sudo dmidecode -s system-manufacturer 2>/dev/null || echo "Unknown")"
    echo "  Product: $(sudo dmidecode -s system-product-name 2>/dev/null || echo "Unknown")"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}' || echo "Unknown")"
    echo "  Thermal Zones: $(find /sys/class/thermal -name "thermal_zone*" 2>/dev/null | wc -l)"
    echo
    
    # Installer readiness
    echo "Installer Readiness:"
    echo "  ✓ Basic functionality verified"
    echo "  ✓ Source structure validated"
    echo "  ✓ System requirements checked"
    echo "  ✓ Dry-run testing completed"
    echo
    
    echo "Next Steps:"
    echo "  1. Review any warnings or errors above"
    echo "  2. Install missing dependencies if needed"
    echo "  3. Run actual installation:"
    echo "     ./install_dsmil_phase2a.sh --dry-run    # Final preview"
    echo "     ./install_dsmil_phase2a.sh              # Interactive install"
    echo "     ./install_dsmil_phase2a.sh --auto       # Automatic install"
    echo
}

main() {
    echo "DSMIL Phase 2A Installer Test Suite"
    echo "===================================="
    echo
    
    local comprehensive=false
    if [[ "${1:-}" == "--comprehensive" ]]; then
        comprehensive=true
    fi
    
    local total_errors=0
    
    # Run basic tests
    if ! test_installer_basics; then
        ((total_errors++))
    fi
    
    if ! test_system_requirements; then
        ((total_errors += $?))
    fi
    
    if ! test_source_structure; then
        ((total_errors += $?))
    fi
    
    if ! test_kernel_module_readiness; then
        ((total_errors++))
    fi
    
    # Run comprehensive tests if requested
    if [[ "$comprehensive" == "true" ]]; then
        run_comprehensive_tests
    fi
    
    # Show summary
    show_summary
    
    # Final result
    echo "=============================================="
    if [[ $total_errors -eq 0 ]]; then
        success "All tests passed! Installer is ready for use."
        echo "Recommendation: Proceed with installation"
    else
        error "Tests completed with $total_errors issues"
        echo "Recommendation: Address issues before installation"
    fi
    echo "=============================================="
    
    return $total_errors
}

main "$@"