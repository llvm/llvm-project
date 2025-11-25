#!/bin/bash
# TPM2 Compatibility Acceleration Library - Production Deployment Script
# Military-grade deployment with security validation and performance optimization
#
# Author: C-INTERNAL Agent
# Date: 2025-09-23
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY

set -euo pipefail

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_NAME="tpm2_compat_accelerated"
readonly VERSION="1.0.0"
readonly LOG_FILE="/var/log/tpm2_acceleration_deploy.log"
readonly BACKUP_DIR="/var/backups/tpm2_acceleration"

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Deployment options
INSTALL_KERNEL_MODULE=false
INSTALL_SYSTEM_WIDE=false
ENABLE_DEBUG=false
SKIP_TESTS=false
BACKUP_EXISTING=true
FORCE_INSTALL=false
DRY_RUN=false

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$@"
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    log "WARN" "$@"
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    log "ERROR" "$@"
    echo -e "${RED}[ERROR]${NC} $*"
}

log_debug() {
    if [[ "$ENABLE_DEBUG" == "true" ]]; then
        log "DEBUG" "$@"
        echo -e "${BLUE}[DEBUG]${NC} $*"
    fi
}

check_root() {
    if [[ $EUID -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

check_dependency() {
    local cmd="$1"
    local package="$2"

    if ! command -v "$cmd" &> /dev/null; then
        log_error "Required dependency '$cmd' not found. Install package: $package"
        return 1
    fi
    return 0
}

create_backup() {
    local source="$1"
    local backup_name="$2"

    if [[ -e "$source" ]] && [[ "$BACKUP_EXISTING" == "true" ]]; then
        local backup_path="${BACKUP_DIR}/${backup_name}_$(date +%Y%m%d_%H%M%S)"

        log_info "Creating backup: $source -> $backup_path"

        if [[ "$DRY_RUN" == "false" ]]; then
            mkdir -p "$BACKUP_DIR"
            cp -r "$source" "$backup_path"
        fi
    fi
}

# =============================================================================
# SYSTEM VALIDATION
# =============================================================================

validate_system() {
    log_info "Validating system requirements..."

    # Check Linux distribution
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot determine Linux distribution"
        return 1
    fi

    local distro=$(grep '^ID=' /etc/os-release | cut -d'=' -f2 | tr -d '"')
    log_info "Detected distribution: $distro"

    # Check kernel version
    local kernel_version=$(uname -r)
    log_info "Kernel version: $kernel_version"

    # Check architecture
    local arch=$(uname -m)
    if [[ "$arch" != "x86_64" ]]; then
        log_warn "Non-x86_64 architecture detected: $arch. Some optimizations may not be available."
    fi
    log_info "Architecture: $arch"

    # Check CPU features
    if [[ -f /proc/cpuinfo ]]; then
        local cpu_features=$(grep '^flags' /proc/cpuinfo | head -1 | cut -d':' -f2)

        if echo "$cpu_features" | grep -q avx2; then
            log_info "✓ AVX2 support detected"
        else
            log_warn "AVX2 support not detected - performance may be reduced"
        fi

        if echo "$cpu_features" | grep -q aes; then
            log_info "✓ AES-NI support detected"
        else
            log_warn "AES-NI support not detected - cryptographic performance may be reduced"
        fi

        # Check for Intel NPU/GNA (simplified detection)
        local cpu_model=$(grep '^model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)
        if echo "$cpu_model" | grep -qi "Core Ultra"; then
            log_info "✓ Intel Core Ultra detected - NPU/GNA acceleration may be available"
        fi
    fi

    return 0
}

check_dependencies() {
    log_info "Checking build dependencies..."

    local missing_deps=()

    # Essential build tools
    if ! check_dependency "gcc" "build-essential"; then
        missing_deps+=("build-essential")
    fi

    if ! check_dependency "make" "make"; then
        missing_deps+=("make")
    fi

    if ! check_dependency "python3" "python3"; then
        missing_deps+=("python3")
    fi

    if ! check_dependency "python3-dev" "python3-dev"; then
        missing_deps+=("python3-dev")
    fi

    # Kernel module dependencies
    if [[ "$INSTALL_KERNEL_MODULE" == "true" ]]; then
        if [[ ! -d "/lib/modules/$(uname -r)/build" ]]; then
            log_error "Kernel headers not found for $(uname -r)"
            missing_deps+=("linux-headers-$(uname -r)")
        fi
    fi

    # Optional performance tools
    if ! command -v "perf" &> /dev/null; then
        log_warn "Performance monitoring tool 'perf' not found. Install linux-tools-$(uname -r) for advanced profiling."
    fi

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Install with: sudo apt-get install ${missing_deps[*]}"
        return 1
    fi

    log_info "✓ All dependencies satisfied"
    return 0
}

# =============================================================================
# BUILD SYSTEM
# =============================================================================

build_library() {
    log_info "Building TPM2 acceleration library..."

    cd "$SCRIPT_DIR"

    # Clean previous build
    if [[ "$DRY_RUN" == "false" ]]; then
        make clean
    fi

    # Set build flags
    local build_flags=""
    if [[ "$ENABLE_DEBUG" == "true" ]]; then
        build_flags="DEBUG=1"
    fi

    # Build library
    log_info "Building with flags: $build_flags"

    if [[ "$DRY_RUN" == "false" ]]; then
        if ! make $build_flags all; then
            log_error "Library build failed"
            return 1
        fi
    fi

    log_info "✓ Library build completed"
    return 0
}

build_kernel_module() {
    if [[ "$INSTALL_KERNEL_MODULE" != "true" ]]; then
        return 0
    fi

    log_info "Building kernel module..."

    cd "$SCRIPT_DIR"

    if [[ "$DRY_RUN" == "false" ]]; then
        if ! make kernel-module; then
            log_error "Kernel module build failed"
            return 1
        fi
    fi

    log_info "✓ Kernel module build completed"
    return 0
}

run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_info "Skipping tests as requested"
        return 0
    fi

    log_info "Running test suite..."

    cd "$SCRIPT_DIR"

    if [[ "$DRY_RUN" == "false" ]]; then
        # Build and run C tests
        if ! make test; then
            log_error "C test build failed"
            return 1
        fi

        if ! make check; then
            log_error "C tests failed"
            return 1
        fi

        # Run Python tests if available
        if [[ -f "tests/comprehensive_test_suite.py" ]]; then
            log_info "Running Python test suite..."

            if ! python3 tests/comprehensive_test_suite.py --quick; then
                log_error "Python tests failed"
                return 1
            fi
        fi
    fi

    log_info "✓ All tests passed"
    return 0
}

# =============================================================================
# INSTALLATION
# =============================================================================

install_library() {
    log_info "Installing TPM2 acceleration library..."

    if [[ "$INSTALL_SYSTEM_WIDE" == "true" ]]; then
        if ! check_root; then
            log_error "System-wide installation requires root privileges"
            return 1
        fi

        # Backup existing installation
        create_backup "/usr/local/lib/libtpm2_compat_accelerated*" "system_libs"
        create_backup "/usr/local/include/tpm2_compat" "system_headers"

        cd "$SCRIPT_DIR"

        if [[ "$DRY_RUN" == "false" ]]; then
            if ! make install; then
                log_error "System-wide installation failed"
                return 1
            fi
        fi

        log_info "✓ System-wide installation completed"
    else
        # Local installation
        local install_dir="$HOME/.local"

        log_info "Installing to local directory: $install_dir"

        if [[ "$DRY_RUN" == "false" ]]; then
            mkdir -p "$install_dir/lib" "$install_dir/include"

            # Copy libraries
            cp lib/libtpm2_compat_accelerated.* "$install_dir/lib/"

            # Copy headers
            cp -r include/tpm2_compat_accelerated.h "$install_dir/include/"
        fi

        log_info "✓ Local installation completed"
        log_info "Add to LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$install_dir/lib:\$LD_LIBRARY_PATH"
    fi

    return 0
}

install_kernel_module() {
    if [[ "$INSTALL_KERNEL_MODULE" != "true" ]]; then
        return 0
    fi

    if ! check_root; then
        log_error "Kernel module installation requires root privileges"
        return 1
    fi

    log_info "Installing kernel module..."

    local module_file="build/tpm2_compat_device.ko"

    if [[ ! -f "$module_file" ]]; then
        log_error "Kernel module not found: $module_file"
        return 1
    fi

    # Check if module is already loaded
    if lsmod | grep -q "tpm2_compat_device"; then
        log_warn "Module already loaded, unloading first..."
        if [[ "$DRY_RUN" == "false" ]]; then
            rmmod tpm2_compat_device || true
        fi
    fi

    # Load module
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! insmod "$module_file"; then
            log_error "Failed to load kernel module"
            return 1
        fi
    fi

    # Verify module loaded
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! lsmod | grep -q "tpm2_compat_device"; then
            log_error "Module failed to load properly"
            return 1
        fi
    fi

    # Check device creation
    if [[ "$DRY_RUN" == "false" ]]; then
        sleep 1  # Give time for device creation

        if [[ ! -e "/dev/tpm0_compat" ]]; then
            log_warn "Device /dev/tpm0_compat not created automatically"
        else
            log_info "✓ Device /dev/tpm0_compat created"
        fi
    fi

    log_info "✓ Kernel module installation completed"
    return 0
}

create_systemd_service() {
    if [[ "$INSTALL_SYSTEM_WIDE" != "true" ]] || ! check_root; then
        return 0
    fi

    log_info "Creating systemd service..."

    local service_file="/etc/systemd/system/tpm2-acceleration.service"

    # Backup existing service
    create_backup "$service_file" "systemd_service"

    if [[ "$DRY_RUN" == "false" ]]; then
        cat > "$service_file" << EOF
[Unit]
Description=TPM2 Acceleration Service
Documentation=man:tpm2_compat_accelerated(8)
After=network.target
Wants=network.target

[Service]
Type=simple
User=tpm2accel
Group=tpm2accel
ExecStart=/usr/local/bin/tpm2_acceleration_service
ExecReload=/bin/kill -HUP \$MAINPID
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=tpm2-acceleration

# Security settings
PrivateTmp=yes
ProtectSystem=strict
ProtectHome=yes
NoNewPrivileges=yes
MemoryDenyWriteExecute=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
LockPersonality=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes

[Install]
WantedBy=multi-user.target
EOF
    fi

    # Create service user
    if [[ "$DRY_RUN" == "false" ]]; then
        if ! id tpm2accel &>/dev/null; then
            useradd -r -s /sbin/nologin -d /var/lib/tpm2accel tpm2accel
            log_info "Created service user: tpm2accel"
        fi

        mkdir -p /var/lib/tpm2accel
        chown tpm2accel:tpm2accel /var/lib/tpm2accel
    fi

    # Reload systemd
    if [[ "$DRY_RUN" == "false" ]]; then
        systemctl daemon-reload
    fi

    log_info "✓ Systemd service created"
    log_info "Start with: sudo systemctl start tpm2-acceleration"
    log_info "Enable on boot: sudo systemctl enable tpm2-acceleration"

    return 0
}

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

validate_installation() {
    log_info "Validating installation..."

    # Check library files
    if [[ "$INSTALL_SYSTEM_WIDE" == "true" ]]; then
        local lib_path="/usr/local/lib"
        local include_path="/usr/local/include"
    else
        local lib_path="$HOME/.local/lib"
        local include_path="$HOME/.local/include"
    fi

    if [[ ! -f "$lib_path/libtpm2_compat_accelerated.so" ]]; then
        log_error "Shared library not found: $lib_path/libtpm2_compat_accelerated.so"
        return 1
    fi

    if [[ ! -f "$lib_path/libtpm2_compat_accelerated.a" ]]; then
        log_error "Static library not found: $lib_path/libtpm2_compat_accelerated.a"
        return 1
    fi

    if [[ ! -f "$include_path/tpm2_compat_accelerated.h" ]]; then
        log_error "Header file not found: $include_path/tpm2_compat_accelerated.h"
        return 1
    fi

    # Test library loading
    if [[ "$DRY_RUN" == "false" ]]; then
        log_info "Testing library loading..."

        if [[ "$INSTALL_SYSTEM_WIDE" != "true" ]]; then
            export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
        fi

        # Test Python bindings
        if python3 -c "
import sys, os
sys.path.insert(0, '$SCRIPT_DIR/src')
try:
    from python_bindings import create_accelerated_library, TPM2LibraryConfig
    print('✓ Python bindings loaded successfully')
except ImportError as e:
    print(f'✗ Python bindings failed: {e}')
    sys.exit(1)
"; then
            log_info "✓ Python bindings validation passed"
        else
            log_error "Python bindings validation failed"
            return 1
        fi
    fi

    # Validate kernel module if installed
    if [[ "$INSTALL_KERNEL_MODULE" == "true" ]] && [[ "$DRY_RUN" == "false" ]]; then
        if lsmod | grep -q "tpm2_compat_device"; then
            log_info "✓ Kernel module loaded successfully"
        else
            log_error "Kernel module not loaded"
            return 1
        fi
    fi

    log_info "✓ Installation validation completed"
    return 0
}

performance_test() {
    if [[ "$SKIP_TESTS" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    log_info "Running performance validation..."

    cd "$SCRIPT_DIR"

    # Run performance benchmarks
    if [[ -f "examples/deployment_example.py" ]]; then
        if python3 examples/deployment_example.py; then
            log_info "✓ Performance validation passed"
        else
            log_warn "Performance validation failed - functionality may still work"
        fi
    fi

    return 0
}

# =============================================================================
# MAIN DEPLOYMENT LOGIC
# =============================================================================

show_usage() {
    cat << EOF
TPM2 Compatibility Acceleration Library - Deployment Script

Usage: $0 [OPTIONS]

Options:
    -k, --kernel-module     Install kernel module (requires root)
    -s, --system-wide       Install system-wide (requires root)
    -d, --debug             Enable debug build and logging
    -t, --skip-tests        Skip test execution
    -n, --no-backup         Skip backup of existing files
    -f, --force             Force installation over existing files
    --dry-run               Show what would be done without executing
    -h, --help              Show this help message

Examples:
    $0                      # Local user installation
    $0 -s                   # System-wide installation
    $0 -s -k                # System-wide with kernel module
    $0 --dry-run -s -k      # Preview system-wide installation

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -k|--kernel-module)
                INSTALL_KERNEL_MODULE=true
                shift
                ;;
            -s|--system-wide)
                INSTALL_SYSTEM_WIDE=true
                shift
                ;;
            -d|--debug)
                ENABLE_DEBUG=true
                shift
                ;;
            -t|--skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            -n|--no-backup)
                BACKUP_EXISTING=false
                shift
                ;;
            -f|--force)
                FORCE_INSTALL=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

main() {
    # Parse command line arguments
    parse_arguments "$@"

    # Show deployment configuration
    log_info "Starting TPM2 Acceleration Library Deployment v$VERSION"
    log_info "Configuration:"
    log_info "  Kernel Module: $INSTALL_KERNEL_MODULE"
    log_info "  System-wide: $INSTALL_SYSTEM_WIDE"
    log_info "  Debug Mode: $ENABLE_DEBUG"
    log_info "  Skip Tests: $SKIP_TESTS"
    log_info "  Backup Existing: $BACKUP_EXISTING"
    log_info "  Dry Run: $DRY_RUN"

    # Create log directory
    if [[ "$DRY_RUN" == "false" ]]; then
        mkdir -p "$(dirname "$LOG_FILE")"
        touch "$LOG_FILE"
    fi

    # System validation
    validate_system || exit 1
    check_dependencies || exit 1

    # Build phase
    build_library || exit 1
    build_kernel_module || exit 1

    # Test phase
    run_tests || exit 1

    # Installation phase
    install_library || exit 1
    install_kernel_module || exit 1
    create_systemd_service || exit 1

    # Validation phase
    validate_installation || exit 1
    performance_test || exit 1

    # Success message
    log_info "✓ TPM2 Acceleration Library deployment completed successfully!"

    if [[ "$INSTALL_SYSTEM_WIDE" == "true" ]]; then
        log_info "System-wide installation complete"
        log_info "Library installed to: /usr/local/lib"
        log_info "Headers installed to: /usr/local/include/tpm2_compat"

        if [[ "$INSTALL_KERNEL_MODULE" == "true" ]]; then
            log_info "Kernel module loaded: tpm2_compat_device"
            log_info "Device available: /dev/tpm0_compat"
        fi
    else
        log_info "Local installation complete"
        log_info "Library installed to: $HOME/.local/lib"
        log_info "Add to environment: export LD_LIBRARY_PATH=$HOME/.local/lib:\$LD_LIBRARY_PATH"
    fi

    log_info "For usage examples, see: examples/deployment_example.py"
    log_info "For documentation, see: README.md"

    return 0
}

# Trap for cleanup on exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        log_error "Deployment failed with exit code $exit_code"
    fi
    exit $exit_code
}

trap cleanup EXIT

# Run main function
main "$@"