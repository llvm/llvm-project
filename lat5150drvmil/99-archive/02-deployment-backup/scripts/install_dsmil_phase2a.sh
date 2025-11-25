#!/bin/bash
#
# DSMIL Phase 2A Expansion System Installer
# Comprehensive installer for chunked IOCTL kernel module and monitoring integration
# 
# Copyright (C) 2025 JRTC1 Educational Development
# Designed for Dell Latitude 5450 MIL-SPEC JRTC1 training variant
#
# VERSION: 2.1.0
# TESTED ON: Linux 6.14.0+ (Ubuntu 24.04+)
# REQUIREMENTS: gcc, make, python3, rust (optional)
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

INSTALLER_VERSION="2.1.0"
INSTALLER_DATE=$(date +"%Y-%m-%d %H:%M:%S")
DSMIL_MODULE_NAME="dsmil-72dev"
DSMIL_VERSION="Phase2A-Enhanced"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_SOURCE_DIR="${SCRIPT_DIR}/01-source/kernel"
MONITORING_DIR="${SCRIPT_DIR}/monitoring"
LOGS_DIR="${SCRIPT_DIR}/logs"
BACKUP_DIR="${SCRIPT_DIR}/backups"

# Installation paths
INSTALL_PREFIX="/opt/dsmil"
SYSTEMD_SERVICE_DIR="/etc/systemd/system"
UDEV_RULES_DIR="/etc/udev/rules.d"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Installation mode flags
QUIET_MODE=false
FORCE_MODE=false
DRY_RUN=false
INSTALL_MONITORING=true
INSTALL_RUST=true
SKIP_VALIDATION=false
AUTO_MODE=false

# =============================================================================
# LOGGING AND OUTPUT FUNCTIONS
# =============================================================================

log() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "${LOGS_DIR}/installer.log"
}

info() {
    echo -e "${BLUE}ℹ${NC} $*" | tee -a "${LOGS_DIR}/installer.log"
}

success() {
    echo -e "${GREEN}✓${NC} $*" | tee -a "${LOGS_DIR}/installer.log"
}

warning() {
    echo -e "${YELLOW}⚠${NC} $*" | tee -a "${LOGS_DIR}/installer.log"
}

error() {
    echo -e "${RED}✗${NC} $*" >&2 | tee -a "${LOGS_DIR}/installer.log"
}

critical() {
    echo -e "${RED}💥 CRITICAL:${NC} $*" >&2 | tee -a "${LOGS_DIR}/installer.log"
}

section() {
    echo
    echo -e "${BOLD}${PURPLE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${PURPLE} $*${NC}"
    echo -e "${BOLD}${PURPLE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo
}

progress() {
    if [[ "$QUIET_MODE" == "false" ]]; then
        local current=$1
        local total=$2
        local desc=$3
        local percent=$((current * 100 / total))
        local bar_length=50
        local filled_length=$((percent * bar_length / 100))
        
        printf "\r${CYAN}[%3d%%]${NC} [" "$percent"
        printf "%${filled_length}s" | tr ' ' '█'
        printf "%$((bar_length - filled_length))s" | tr ' ' '░'
        printf "] %s" "$desc"
        
        if [[ $current -eq $total ]]; then
            echo
        fi
    fi
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

create_directories() {
    local dirs=("$@")
    for dir in "${dirs[@]}"; do
        if [[ "$DRY_RUN" == "true" ]]; then
            info "[DRY RUN] Would create directory: $dir"
        else
            mkdir -p "$dir"
            success "Created directory: $dir"
        fi
    done
}

backup_file() {
    local file="$1"
    local backup_suffix="${2:-$(date +%Y%m%d_%H%M%S)}"
    
    if [[ -f "$file" ]]; then
        local backup_file="${file}.backup_${backup_suffix}"
        if [[ "$DRY_RUN" == "true" ]]; then
            info "[DRY RUN] Would backup $file to $backup_file"
        else
            cp "$file" "$backup_file"
            success "Backed up $file to $backup_file"
        fi
    fi
}

check_command() {
    local cmd="$1"
    local required="${2:-true}"
    
    if command -v "$cmd" >/dev/null 2>&1; then
        success "Command available: $cmd ($(command -v "$cmd"))"
        return 0
    else
        if [[ "$required" == "true" ]]; then
            error "Required command missing: $cmd"
            return 1
        else
            warning "Optional command missing: $cmd"
            return 1
        fi
    fi
}

check_kernel_version() {
    local kernel_version
    kernel_version=$(uname -r)
    local major_version
    major_version=$(echo "$kernel_version" | cut -d. -f1)
    local minor_version
    minor_version=$(echo "$kernel_version" | cut -d. -f2)
    
    info "Detected kernel version: $kernel_version"
    
    if [[ $major_version -lt 6 ]] || [[ $major_version -eq 6 && $minor_version -lt 14 ]]; then
        warning "Kernel version $kernel_version is below recommended 6.14.0+"
        if [[ "$FORCE_MODE" == "false" ]]; then
            error "Use --force to bypass kernel version check"
            return 1
        fi
    else
        success "Kernel version $kernel_version meets requirements"
    fi
    
    return 0
}

detect_platform() {
    local platform="unknown"
    
    if [[ -f /etc/os-release ]]; then
        source /etc/os-release
        platform="$ID-$VERSION_ID"
    elif [[ -f /etc/debian_version ]]; then
        platform="debian-$(cat /etc/debian_version)"
    elif [[ -f /etc/redhat-release ]]; then
        platform="rhel-$(rpm -qa \*-release | grep -Ei "redhat|centos" | cut -d"-" -f3)"
    fi
    
    info "Detected platform: $platform"
    echo "$platform"
}

# =============================================================================
# SYSTEM VALIDATION FUNCTIONS
# =============================================================================

validate_system_requirements() {
    section "VALIDATING SYSTEM REQUIREMENTS"
    
    local validation_errors=0
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        warning "Running as root. Consider using sudo for specific commands only."
    fi
    
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        info "Testing sudo access (password may be required)..."
        if ! sudo true; then
            error "Sudo access required for installation"
            ((validation_errors++))
        else
            success "Sudo access confirmed"
        fi
    else
        success "Passwordless sudo access available"
    fi
    
    # Check kernel version
    if ! check_kernel_version; then
        ((validation_errors++))
    fi
    
    # Check required commands
    local required_commands=("gcc" "make" "python3" "modprobe" "dmesg" "lsmod")
    for cmd in "${required_commands[@]}"; do
        if ! check_command "$cmd"; then
            ((validation_errors++))
        fi
    done
    
    # Check optional commands
    local optional_commands=("cargo" "rustc" "systemctl" "udevadm")
    for cmd in "${optional_commands[@]}"; do
        check_command "$cmd" false
    done
    
    # Check kernel headers
    local kernel_headers_path="/lib/modules/$(uname -r)/build"
    if [[ -d "$kernel_headers_path" ]]; then
        success "Kernel headers available: $kernel_headers_path"
    else
        error "Kernel headers not found. Install linux-headers-$(uname -r)"
        ((validation_errors++))
    fi
    
    # Check hardware compatibility
    validate_hardware_compatibility
    
    if [[ $validation_errors -gt 0 ]]; then
        error "System validation failed with $validation_errors errors"
        if [[ "$FORCE_MODE" == "false" ]]; then
            critical "Use --force to bypass validation errors (NOT RECOMMENDED)"
            return 1
        fi
        warning "Continuing with --force mode despite validation errors"
    else
        success "System validation completed successfully"
    fi
    
    return 0
}

validate_hardware_compatibility() {
    info "Validating hardware compatibility..."
    
    # Check for Dell hardware
    local dmi_vendor
    dmi_vendor=$(sudo dmidecode -s system-manufacturer 2>/dev/null || echo "Unknown")
    
    if [[ "$dmi_vendor" == *"Dell"* ]]; then
        success "Dell hardware detected: $dmi_vendor"
        
        # Check specific model
        local dmi_product
        dmi_product=$(sudo dmidecode -s system-product-name 2>/dev/null || echo "Unknown")
        info "Product: $dmi_product"
        
        if [[ "$dmi_product" == *"Latitude 5450"* ]]; then
            success "Target hardware confirmed: Dell Latitude 5450"
        else
            warning "Hardware model not specifically tested: $dmi_product"
        fi
    else
        warning "Non-Dell hardware detected: $dmi_vendor"
        warning "DSMIL functionality may be limited on non-Dell systems"
    fi
    
    # Check for SMBIOS support
    if [[ -d /sys/firmware/dmi ]]; then
        success "SMBIOS/DMI interface available"
    else
        warning "SMBIOS/DMI interface not detected"
    fi
    
    # Check thermal zones
    local thermal_zones
    thermal_zones=$(find /sys/class/thermal -name "thermal_zone*" 2>/dev/null | wc -l)
    if [[ $thermal_zones -gt 0 ]]; then
        success "Thermal monitoring available ($thermal_zones zones)"
    else
        warning "No thermal zones detected"
    fi
}

# =============================================================================
# DEPENDENCY INSTALLATION FUNCTIONS
# =============================================================================

install_system_dependencies() {
    section "INSTALLING SYSTEM DEPENDENCIES"
    
    local platform
    platform=$(detect_platform)
    
    case "$platform" in
        ubuntu-* | debian-*)
            install_debian_dependencies
            ;;
        fedora-* | centos-* | rhel-*)
            install_redhat_dependencies
            ;;
        *)
            warning "Unknown platform: $platform"
            warning "Manual dependency installation may be required"
            ;;
    esac
}

install_debian_dependencies() {
    info "Installing dependencies for Debian/Ubuntu..."
    
    local packages=(
        "build-essential"
        "linux-headers-$(uname -r)"
        "python3"
        "python3-pip"
        "python3-dev"
        "dkms"
        "git"
    )
    
    if [[ "$INSTALL_RUST" == "true" ]]; then
        packages+=("cargo" "rustc")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install packages: ${packages[*]}"
    else
        sudo apt-get update
        sudo apt-get install -y "${packages[@]}"
        success "System dependencies installed"
    fi
}

install_redhat_dependencies() {
    info "Installing dependencies for Red Hat/Fedora/CentOS..."
    
    local packages=(
        "gcc"
        "make"
        "kernel-devel"
        "kernel-headers"
        "python3"
        "python3-pip"
        "python3-devel"
        "dkms"
        "git"
    )
    
    if [[ "$INSTALL_RUST" == "true" ]]; then
        packages+=("cargo" "rust")
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install packages: ${packages[*]}"
    else
        if command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y "${packages[@]}"
        elif command -v yum >/dev/null 2>&1; then
            sudo yum install -y "${packages[@]}"
        else
            error "No package manager found (dnf/yum)"
            return 1
        fi
        success "System dependencies installed"
    fi
}

install_python_dependencies() {
    section "INSTALLING PYTHON DEPENDENCIES"
    
    local python_packages=(
        "psutil>=5.8.0"
        "fcntl2"
        "struct"
        "dataclasses"
        "pathlib"
        "datetime"
        "json"
        "subprocess"
        "time"
        "os"
        "sys"
        "signal"
        "threading"
        "logging"
        "argparse"
    )
    
    info "Installing Python packages for chunked IOCTL system..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install Python packages: ${python_packages[*]}"
    else
        for package in "${python_packages[@]}"; do
            # Skip built-in modules
            if [[ "$package" =~ ^(struct|dataclasses|pathlib|datetime|json|subprocess|time|os|sys|signal|threading|logging|argparse)$ ]]; then
                info "Skipping built-in module: $package"
                continue
            fi
            
            progress $((++current)) ${#python_packages[@]} "Installing $package"
            
            if python3 -c "import ${package%%>=*}" 2>/dev/null; then
                success "Package already available: $package"
            else
                if pip3 install --user "$package"; then
                    success "Installed Python package: $package"
                else
                    warning "Failed to install Python package: $package"
                fi
            fi
        done
    fi
}

# =============================================================================
# KERNEL MODULE FUNCTIONS
# =============================================================================

build_kernel_module() {
    section "BUILDING DSMIL KERNEL MODULE"
    
    if [[ ! -d "$KERNEL_SOURCE_DIR" ]]; then
        error "Kernel source directory not found: $KERNEL_SOURCE_DIR"
        return 1
    fi
    
    cd "$KERNEL_SOURCE_DIR"
    
    # Create backup of existing module if present
    if [[ -f "${DSMIL_MODULE_NAME}.ko" ]]; then
        backup_file "${DSMIL_MODULE_NAME}.ko"
    fi
    
    # Clean previous builds
    info "Cleaning previous build artifacts..."
    if [[ "$DRY_RUN" == "false" ]]; then
        make clean >/dev/null 2>&1 || true
    fi
    
    # Build Rust components if enabled
    if [[ "$INSTALL_RUST" == "true" && -d "rust" ]]; then
        info "Building Rust safety components..."
        if [[ "$DRY_RUN" == "false" ]]; then
            if ! make rust-lib; then
                error "Rust build failed"
                if [[ "$FORCE_MODE" == "false" ]]; then
                    return 1
                fi
                warning "Continuing without Rust components"
                INSTALL_RUST=false
            else
                success "Rust components built successfully"
            fi
        fi
    fi
    
    # Build kernel module
    info "Building kernel module..."
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would build kernel module with: make all"
    else
        if make all; then
            success "Kernel module built successfully"
            
            # Verify module was created
            if [[ -f "${DSMIL_MODULE_NAME}.ko" ]]; then
                local module_size
                module_size=$(stat -c%s "${DSMIL_MODULE_NAME}.ko")
                success "Module created: ${DSMIL_MODULE_NAME}.ko (${module_size} bytes)"
                
                # Get module info
                modinfo "${DSMIL_MODULE_NAME}.ko" | head -10
            else
                error "Module file not created: ${DSMIL_MODULE_NAME}.ko"
                return 1
            fi
        else
            error "Kernel module build failed"
            return 1
        fi
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

install_kernel_module() {
    section "INSTALLING KERNEL MODULE"
    
    cd "$KERNEL_SOURCE_DIR"
    
    # Check if module exists
    if [[ ! -f "${DSMIL_MODULE_NAME}.ko" ]]; then
        error "Kernel module not found: ${DSMIL_MODULE_NAME}.ko"
        return 1
    fi
    
    # Remove existing module if loaded
    if lsmod | grep -q "$DSMIL_MODULE_NAME"; then
        info "Removing existing module..."
        if [[ "$DRY_RUN" == "false" ]]; then
            sudo rmmod "$DSMIL_MODULE_NAME" || warning "Could not remove existing module"
        fi
    fi
    
    # Install module to system location
    local install_path="/lib/modules/$(uname -r)/extra"
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would install module to: $install_path"
        info "[DRY RUN] Would run: sudo depmod -a"
        info "[DRY RUN] Would load module with: sudo modprobe $DSMIL_MODULE_NAME"
    else
        sudo mkdir -p "$install_path"
        sudo cp "${DSMIL_MODULE_NAME}.ko" "$install_path/"
        sudo depmod -a
        success "Module installed to system"
        
        # Load the module
        info "Loading kernel module..."
        if sudo modprobe "$DSMIL_MODULE_NAME"; then
            success "Kernel module loaded successfully"
            
            # Verify module is loaded
            if lsmod | grep -q "$DSMIL_MODULE_NAME"; then
                success "Module verified in kernel"
                
                # Show module information
                info "Module information:"
                lsmod | grep "$DSMIL_MODULE_NAME"
                
                # Check for device files
                local device_count=0
                for dev in /dev/dsmil*; do
                    if [[ -e "$dev" ]]; then
                        success "Device file created: $dev"
                        ((device_count++))
                    fi
                done
                
                if [[ $device_count -eq 0 ]]; then
                    warning "No device files found in /dev/"
                fi
            else
                error "Module not found in kernel after loading"
                return 1
            fi
        else
            error "Failed to load kernel module"
            return 1
        fi
    fi
    
    cd "$SCRIPT_DIR"
    return 0
}

# =============================================================================
# MONITORING SYSTEM FUNCTIONS
# =============================================================================

install_monitoring_system() {
    section "INSTALLING MONITORING SYSTEM"
    
    if [[ "$INSTALL_MONITORING" == "false" ]]; then
        info "Monitoring system installation skipped"
        return 0
    fi
    
    if [[ ! -d "$MONITORING_DIR" ]]; then
        error "Monitoring directory not found: $MONITORING_DIR"
        return 1
    fi
    
    # Install monitoring scripts to system location
    local monitor_install_dir="$INSTALL_PREFIX/monitoring"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create monitoring directory: $monitor_install_dir"
        info "[DRY RUN] Would copy monitoring scripts to system location"
    else
        sudo mkdir -p "$monitor_install_dir"
        sudo cp -r "$MONITORING_DIR"/* "$monitor_install_dir/"
        sudo chmod +x "$monitor_install_dir"/*.sh
        sudo chmod +x "$monitor_install_dir"/*.py
        success "Monitoring scripts installed to $monitor_install_dir"
    fi
    
    # Create monitoring service
    create_monitoring_service
    
    # Create udev rules for device permissions
    create_udev_rules
    
    # Verify monitoring system
    verify_monitoring_installation
    
    return 0
}

create_monitoring_service() {
    info "Creating systemd service for monitoring..."
    
    local service_content="[Unit]
Description=DSMIL Monitoring Service
Documentation=file://$INSTALL_PREFIX/docs/monitoring.md
After=multi-user.target
Wants=network.target

[Service]
Type=forking
ExecStart=$INSTALL_PREFIX/monitoring/start_monitoring_session.sh --daemon
ExecStop=$INSTALL_PREFIX/monitoring/emergency_stop.sh
Restart=on-failure
RestartSec=10
User=root
Group=root
StandardOutput=journal
StandardError=journal

# Security settings
PrivateNetwork=false
PrivateDevices=false
ProtectSystem=false
ProtectHome=false
NoNewPrivileges=false

# Environment
Environment=DSMIL_INSTALL_PREFIX=$INSTALL_PREFIX
Environment=DSMIL_MODULE_NAME=$DSMIL_MODULE_NAME

[Install]
WantedBy=multi-user.target
"
    
    local service_file="$SYSTEMD_SERVICE_DIR/dsmil-monitor.service"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create systemd service: $service_file"
    else
        echo "$service_content" | sudo tee "$service_file" > /dev/null
        sudo systemctl daemon-reload
        success "Systemd service created: $service_file"
    fi
}

create_udev_rules() {
    info "Creating udev rules for device permissions..."
    
    local udev_content="# DSMIL device permissions
# Allow access to DSMIL devices for users in dsmil group
SUBSYSTEM==\"misc\", KERNEL==\"dsmil*\", GROUP=\"dsmil\", MODE=\"0664\"
SUBSYSTEM==\"char\", KERNEL==\"dsmil*\", GROUP=\"dsmil\", MODE=\"0664\"

# Additional rules for Dell SMBIOS access
SUBSYSTEM==\"firmware\", ATTRS{family}==\"dell_wmi_sysman\", GROUP=\"dsmil\", MODE=\"0664\"
"
    
    local udev_file="$UDEV_RULES_DIR/99-dsmil.rules"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create udev rules: $udev_file"
        info "[DRY RUN] Would create dsmil group"
        info "[DRY RUN] Would reload udev rules"
    else
        echo "$udev_content" | sudo tee "$udev_file" > /dev/null
        
        # Create dsmil group
        if ! getent group dsmil >/dev/null; then
            sudo groupadd dsmil
            success "Created dsmil group"
        fi
        
        # Add current user to dsmil group
        sudo usermod -a -G dsmil "$USER"
        info "Added $USER to dsmil group (logout/login required)"
        
        # Reload udev rules
        sudo udevadm control --reload-rules
        sudo udevadm trigger
        
        success "Udev rules installed and activated"
    fi
}

verify_monitoring_installation() {
    info "Verifying monitoring system installation..."
    
    local monitor_script="$INSTALL_PREFIX/monitoring/dsmil_comprehensive_monitor.py"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would verify monitoring installation"
        return 0
    fi
    
    if [[ -f "$monitor_script" ]]; then
        success "Main monitoring script found: $monitor_script"
        
        # Test monitoring script
        if python3 "$monitor_script" --help >/dev/null 2>&1; then
            success "Monitoring script executable and functional"
        else
            warning "Monitoring script may have issues"
        fi
    else
        error "Main monitoring script not found: $monitor_script"
        return 1
    fi
    
    return 0
}

# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

create_configuration_files() {
    section "CREATING CONFIGURATION FILES"
    
    local config_dir="$INSTALL_PREFIX/config"
    
    if [[ "$DRY_RUN" == "false" ]]; then
        sudo mkdir -p "$config_dir"
    fi
    
    # Main DSMIL configuration
    create_main_config
    
    # Monitoring configuration
    create_monitoring_config
    
    # Safety configuration
    create_safety_config
    
    success "Configuration files created"
}

create_main_config() {
    local config_content="{
    \"dsmil\": {
        \"version\": \"$DSMIL_VERSION\",
        \"module_name\": \"$DSMIL_MODULE_NAME\",
        \"device_count\": 108,
        \"chunked_ioctl\": {
            \"enabled\": true,
            \"chunk_size\": 256,
            \"devices_per_chunk\": 5,
            \"max_chunks\": 22
        },
        \"quarantined_devices\": [
            \"0x8009\", \"0x800A\", \"0x800B\", 
            \"0x8019\", \"0x8029\"
        ]
    },
    \"system\": {
        \"install_prefix\": \"$INSTALL_PREFIX\",
        \"log_level\": \"INFO\",
        \"max_log_size\": \"10MB\",
        \"backup_count\": 5
    },
    \"hardware\": {
        \"platform\": \"$(detect_platform)\",
        \"kernel_version\": \"$(uname -r)\",
        \"installation_date\": \"$INSTALLER_DATE\"
    }
}"
    
    local config_file="$INSTALL_PREFIX/config/dsmil.json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create main config: $config_file"
    else
        echo "$config_content" | sudo tee "$config_file" > /dev/null
        success "Main configuration created: $config_file"
    fi
}

create_monitoring_config() {
    local config_content="{
    \"monitoring\": {
        \"enabled\": true,
        \"update_interval\": 2,
        \"modes\": [\"dashboard\", \"resources\", \"tokens\", \"alerts\"],
        \"auto_start\": false
    },
    \"thresholds\": {
        \"temperature\": {
            \"warning\": 85,
            \"critical\": 90,
            \"emergency\": 95
        },
        \"memory\": {
            \"warning\": 80,
            \"critical\": 90,
            \"emergency\": 95
        },
        \"cpu\": {
            \"warning\": 80,
            \"critical\": 90,
            \"emergency\": 95
        }
    },
    \"token_ranges\": {
        \"primary_range\": \"0x0480-0x04C7\",
        \"test_ranges\": [
            \"0x0400-0x0447\",
            \"0x0500-0x0547\"
        ]
    }
}"
    
    local config_file="$INSTALL_PREFIX/config/monitoring.json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create monitoring config: $config_file"
    else
        echo "$config_content" | sudo tee "$config_file" > /dev/null
        success "Monitoring configuration created: $config_file"
    fi
}

create_safety_config() {
    local config_content="{
    \"safety\": {
        \"enabled\": true,
        \"emergency_stop_timeout\": 5,
        \"max_test_duration\": 300,
        \"require_confirmation\": true,
        \"dry_run_default\": true
    },
    \"emergency_procedures\": {
        \"thermal_emergency\": [
            \"stop_all_testing\",
            \"unload_module\",
            \"log_emergency\",
            \"alert_admin\"
        ],
        \"memory_emergency\": [
            \"stop_testing\",
            \"cleanup_resources\",
            \"log_state\",
            \"restart_monitoring\"
        ]
    },
    \"backup\": {
        \"auto_backup\": true,
        \"backup_interval\": 3600,
        \"max_backups\": 10,
        \"backup_location\": \"$BACKUP_DIR\"
    }
}"
    
    local config_file="$INSTALL_PREFIX/config/safety.json"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create safety config: $config_file"
    else
        echo "$config_content" | sudo tee "$config_file" > /dev/null
        success "Safety configuration created: $config_file"
    fi
}

# =============================================================================
# VERIFICATION AND TESTING FUNCTIONS
# =============================================================================

run_installation_tests() {
    section "RUNNING INSTALLATION VERIFICATION TESTS"
    
    local test_failures=0
    
    # Test 1: Module loading
    if ! test_module_loading; then
        ((test_failures++))
    fi
    
    # Test 2: Device file creation
    if ! test_device_files; then
        ((test_failures++))
    fi
    
    # Test 3: Basic IOCTL functionality
    if ! test_basic_ioctl; then
        ((test_failures++))
    fi
    
    # Test 4: Monitoring system
    if [[ "$INSTALL_MONITORING" == "true" ]]; then
        if ! test_monitoring_system; then
            ((test_failures++))
        fi
    fi
    
    # Test 5: Configuration files
    if ! test_configuration_files; then
        ((test_failures++))
    fi
    
    # Test 6: Chunked IOCTL system
    if ! test_chunked_ioctl; then
        ((test_failures++))
    fi
    
    if [[ $test_failures -eq 0 ]]; then
        success "All installation tests passed"
        return 0
    else
        error "Installation tests failed: $test_failures errors"
        return 1
    fi
}

test_module_loading() {
    info "Testing kernel module loading..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test module loading"
        return 0
    fi
    
    if lsmod | grep -q "$DSMIL_MODULE_NAME"; then
        success "Module loaded: $DSMIL_MODULE_NAME"
        
        # Check module parameters
        local module_info
        module_info=$(modinfo "$DSMIL_MODULE_NAME" 2>/dev/null)
        if [[ -n "$module_info" ]]; then
            info "Module info available"
            echo "$module_info" | grep -E "(version|description|author)" || true
        fi
        
        return 0
    else
        error "Module not loaded: $DSMIL_MODULE_NAME"
        return 1
    fi
}

test_device_files() {
    info "Testing device file creation..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test device files"
        return 0
    fi
    
    local device_found=false
    
    for dev in /dev/dsmil*; do
        if [[ -e "$dev" ]]; then
            success "Device file exists: $dev"
            
            # Check permissions
            local perms
            perms=$(stat -c "%A %U:%G" "$dev")
            info "Device permissions: $dev ($perms)"
            
            device_found=true
        fi
    done
    
    if [[ "$device_found" == "true" ]]; then
        return 0
    else
        error "No DSMIL device files found"
        return 1
    fi
}

test_basic_ioctl() {
    info "Testing basic IOCTL functionality..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test basic IOCTL"
        return 0
    fi
    
    # Create a simple test program
    local test_program=$(mktemp)
    
    cat > "${test_program}.c" << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>

#define MILDEV_IOC_GET_VERSION _IOR('M', 1, uint32_t)

int main() {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        printf("Cannot open device\n");
        return 1;
    }
    
    uint32_t version = 0;
    if (ioctl(fd, MILDEV_IOC_GET_VERSION, &version) == 0) {
        printf("Version: 0x%08X\n", version);
        close(fd);
        return 0;
    } else {
        printf("IOCTL failed\n");
        close(fd);
        return 1;
    }
}
EOF
    
    if gcc -o "$test_program" "${test_program}.c" 2>/dev/null; then
        if "$test_program" 2>/dev/null; then
            success "Basic IOCTL test passed"
            rm -f "$test_program" "${test_program}.c"
            return 0
        else
            warning "Basic IOCTL test failed (device may not be ready)"
            rm -f "$test_program" "${test_program}.c"
            return 1
        fi
    else
        warning "Could not compile IOCTL test program"
        rm -f "${test_program}.c"
        return 1
    fi
}

test_monitoring_system() {
    info "Testing monitoring system..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test monitoring system"
        return 0
    fi
    
    local monitor_script="$INSTALL_PREFIX/monitoring/dsmil_comprehensive_monitor.py"
    
    if [[ -f "$monitor_script" ]]; then
        # Test that the script can run
        if timeout 5 python3 "$monitor_script" --help >/dev/null 2>&1; then
            success "Monitoring script functional"
            return 0
        else
            error "Monitoring script test failed"
            return 1
        fi
    else
        error "Monitoring script not found"
        return 1
    fi
}

test_configuration_files() {
    info "Testing configuration files..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test configuration files"
        return 0
    fi
    
    local config_files=(
        "$INSTALL_PREFIX/config/dsmil.json"
        "$INSTALL_PREFIX/config/monitoring.json"
        "$INSTALL_PREFIX/config/safety.json"
    )
    
    local config_errors=0
    
    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            # Test JSON validity
            if python3 -c "import json; json.load(open('$config_file'))" 2>/dev/null; then
                success "Configuration valid: $(basename "$config_file")"
            else
                error "Configuration invalid: $config_file"
                ((config_errors++))
            fi
        else
            error "Configuration missing: $config_file"
            ((config_errors++))
        fi
    done
    
    if [[ $config_errors -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

test_chunked_ioctl() {
    info "Testing chunked IOCTL system..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would test chunked IOCTL system"
        return 0
    fi
    
    # Test if chunked IOCTL test script exists and runs
    local test_script="$SCRIPT_DIR/test_chunked_ioctl.py"
    
    if [[ -f "$test_script" ]]; then
        if timeout 30 python3 "$test_script" --dry-run 2>/dev/null; then
            success "Chunked IOCTL system test passed"
            return 0
        else
            warning "Chunked IOCTL test failed or timed out"
            return 1
        fi
    else
        warning "Chunked IOCTL test script not found"
        return 1
    fi
}

# =============================================================================
# ROLLBACK AND CLEANUP FUNCTIONS
# =============================================================================

create_rollback_script() {
    info "Creating rollback script..."
    
    local rollback_script="$INSTALL_PREFIX/bin/rollback_dsmil.sh"
    local rollback_content="#!/bin/bash
# DSMIL Installation Rollback Script
# Generated by installer on $INSTALLER_DATE

set -euo pipefail

echo \"DSMIL Installation Rollback\"
echo \"============================\"

# Stop and disable services
if systemctl is-active --quiet dsmil-monitor; then
    echo \"Stopping DSMIL monitor service...\"
    sudo systemctl stop dsmil-monitor
fi

if systemctl is-enabled --quiet dsmil-monitor; then
    echo \"Disabling DSMIL monitor service...\"
    sudo systemctl disable dsmil-monitor
fi

# Remove systemd service
if [[ -f '$SYSTEMD_SERVICE_DIR/dsmil-monitor.service' ]]; then
    echo \"Removing systemd service...\"
    sudo rm -f '$SYSTEMD_SERVICE_DIR/dsmil-monitor.service'
    sudo systemctl daemon-reload
fi

# Unload kernel module
if lsmod | grep -q '$DSMIL_MODULE_NAME'; then
    echo \"Unloading kernel module...\"
    sudo rmmod '$DSMIL_MODULE_NAME' || echo \"Warning: Could not unload module\"
fi

# Remove kernel module from system
MODULE_PATH=\"/lib/modules/\$(uname -r)/extra/${DSMIL_MODULE_NAME}.ko\"
if [[ -f \"\$MODULE_PATH\" ]]; then
    echo \"Removing kernel module from system...\"
    sudo rm -f \"\$MODULE_PATH\"
    sudo depmod -a
fi

# Remove udev rules
if [[ -f '$UDEV_RULES_DIR/99-dsmil.rules' ]]; then
    echo \"Removing udev rules...\"
    sudo rm -f '$UDEV_RULES_DIR/99-dsmil.rules'
    sudo udevadm control --reload-rules
fi

# Remove installation directory
if [[ -d '$INSTALL_PREFIX' ]]; then
    echo \"Removing installation directory...\"
    sudo rm -rf '$INSTALL_PREFIX'
fi

# Remove dsmil group
if getent group dsmil >/dev/null; then
    echo \"Removing dsmil group...\"
    sudo groupdel dsmil || echo \"Warning: Could not remove dsmil group\"
fi

echo \"Rollback completed successfully\"
echo \"Note: Backup files in $BACKUP_DIR are preserved\"
"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        info "[DRY RUN] Would create rollback script: $rollback_script"
    else
        sudo mkdir -p "$(dirname "$rollback_script")"
        echo "$rollback_content" | sudo tee "$rollback_script" > /dev/null
        sudo chmod +x "$rollback_script"
        success "Rollback script created: $rollback_script"
    fi
}

cleanup_installation() {
    info "Cleaning up installation files..."
    
    # Remove temporary files
    rm -f /tmp/dsmil_installer_*
    
    # Clean build directory
    if [[ -d "$KERNEL_SOURCE_DIR" ]]; then
        cd "$KERNEL_SOURCE_DIR"
        make clean >/dev/null 2>&1 || true
        cd "$SCRIPT_DIR"
    fi
    
    success "Installation cleanup completed"
}

# =============================================================================
# MAIN INSTALLATION FUNCTIONS
# =============================================================================

show_usage() {
    cat << EOF
DSMIL Phase 2A Expansion System Installer v$INSTALLER_VERSION

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -q, --quiet             Quiet mode (minimal output)
    -f, --force             Force installation (bypass validation)
    -n, --dry-run           Dry run (show what would be done)
    --no-monitoring         Skip monitoring system installation
    --no-rust              Skip Rust components
    --skip-validation       Skip system validation
    --auto                  Automatic mode (no prompts)

EXAMPLES:
    $0                      # Interactive installation
    $0 --dry-run            # Preview installation steps
    $0 --force --auto       # Force automatic installation
    $0 --no-monitoring      # Install without monitoring system

DESCRIPTION:
    This installer sets up the DSMIL Phase 2A expansion system with:
    - Chunked IOCTL kernel module for large structure transfers
    - Comprehensive monitoring and safety systems
    - Cross-platform compatibility for Linux 6.14.0+
    - Integration with existing DSMIL infrastructure

REQUIREMENTS:
    - Linux 6.14.0+ kernel
    - GCC compiler and kernel headers
    - Python 3.6+ with development headers
    - Sudo access for system installation
    - Dell Latitude 5450 MIL-SPEC (recommended)

SAFETY:
    - All operations include rollback capabilities
    - Dry-run mode available for testing
    - Comprehensive validation before installation
    - Emergency stop procedures included

For more information, see: docs/PHASE2_CHUNKED_IOCTL_SOLUTION.md
EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -q|--quiet)
                QUIET_MODE=true
                shift
                ;;
            -f|--force)
                FORCE_MODE=true
                shift
                ;;
            -n|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-monitoring)
                INSTALL_MONITORING=false
                shift
                ;;
            --no-rust)
                INSTALL_RUST=false
                shift
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --auto)
                AUTO_MODE=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

show_installation_summary() {
    section "INSTALLATION SUMMARY"
    
    cat << EOF
DSMIL Phase 2A Expansion System
Version: $DSMIL_VERSION
Installer: v$INSTALLER_VERSION
Date: $INSTALLER_DATE
Platform: $(detect_platform)
Kernel: $(uname -r)

INSTALLATION OPTIONS:
$(if [[ "$DRY_RUN" == "true" ]]; then echo "  Mode: DRY RUN (no changes will be made)"; else echo "  Mode: LIVE INSTALLATION"; fi)
$(if [[ "$FORCE_MODE" == "true" ]]; then echo "  Force: Enabled (bypassing validation)"; else echo "  Force: Disabled"; fi)
$(if [[ "$INSTALL_MONITORING" == "true" ]]; then echo "  Monitoring: Enabled"; else echo "  Monitoring: Disabled"; fi)
$(if [[ "$INSTALL_RUST" == "true" ]]; then echo "  Rust Components: Enabled"; else echo "  Rust Components: Disabled"; fi)

COMPONENTS TO INSTALL:
  ✓ Chunked IOCTL kernel module ($DSMIL_MODULE_NAME)
  ✓ System dependencies and build tools
  ✓ Python dependencies for chunked transfers
$(if [[ "$INSTALL_RUST" == "true" ]]; then echo "  ✓ Rust safety components"; fi)
$(if [[ "$INSTALL_MONITORING" == "true" ]]; then echo "  ✓ Comprehensive monitoring system"; fi)
  ✓ Configuration files and safety procedures
  ✓ Device permissions and udev rules
  ✓ Installation verification tests

INSTALLATION PATHS:
  Install Prefix: $INSTALL_PREFIX
  Kernel Module: /lib/modules/$(uname -r)/extra/
  Service Files: $SYSTEMD_SERVICE_DIR
  Device Rules: $UDEV_RULES_DIR
  Log Files: $LOGS_DIR
  Backups: $BACKUP_DIR

WARNING: This installer will:
  - Install kernel modules (requires sudo)
  - Modify system configuration files
  - Create system users and groups
  - Install systemd services
  - Load kernel modules with hardware access

EOF

    if [[ "$AUTO_MODE" == "false" && "$DRY_RUN" == "false" ]]; then
        echo -n "Continue with installation? [y/N]: "
        read -r response
        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            info "Installation cancelled by user"
            exit 0
        fi
    fi
    
    echo
}

main_installation() {
    local step=1
    local total_steps=10
    
    # Step 1: System validation
    if [[ "$SKIP_VALIDATION" == "false" ]]; then
        progress $step $total_steps "Validating system requirements"
        validate_system_requirements
        ((step++))
    fi
    
    # Step 2: Create directories
    progress $step $total_steps "Creating installation directories"
    create_directories "$INSTALL_PREFIX" "$LOGS_DIR" "$BACKUP_DIR" "$INSTALL_PREFIX/bin" "$INSTALL_PREFIX/config" "$INSTALL_PREFIX/docs"
    ((step++))
    
    # Step 3: Install system dependencies
    progress $step $total_steps "Installing system dependencies"
    install_system_dependencies
    ((step++))
    
    # Step 4: Install Python dependencies
    progress $step $total_steps "Installing Python dependencies"
    install_python_dependencies
    ((step++))
    
    # Step 5: Build kernel module
    progress $step $total_steps "Building kernel module"
    build_kernel_module
    ((step++))
    
    # Step 6: Install kernel module
    progress $step $total_steps "Installing kernel module"
    install_kernel_module
    ((step++))
    
    # Step 7: Install monitoring system
    if [[ "$INSTALL_MONITORING" == "true" ]]; then
        progress $step $total_steps "Installing monitoring system"
        install_monitoring_system
    else
        progress $step $total_steps "Skipping monitoring system"
    fi
    ((step++))
    
    # Step 8: Create configuration files
    progress $step $total_steps "Creating configuration files"
    create_configuration_files
    ((step++))
    
    # Step 9: Run verification tests
    progress $step $total_steps "Running verification tests"
    run_installation_tests
    ((step++))
    
    # Step 10: Finalize installation
    progress $step $total_steps "Finalizing installation"
    create_rollback_script
    cleanup_installation
    ((step++))
    
    echo
}

show_completion_message() {
    section "INSTALLATION COMPLETE"
    
    cat << EOF
🎉 DSMIL Phase 2A Expansion System installation completed successfully!

WHAT'S INSTALLED:
  ✅ Kernel Module: $DSMIL_MODULE_NAME (loaded and ready)
  ✅ Device Files: /dev/dsmil* (permissions configured)
  ✅ Monitoring System: $INSTALL_PREFIX/monitoring/
$(if [[ "$INSTALL_RUST" == "true" ]]; then echo "  ✅ Rust Safety Components: Integrated"; fi)
  ✅ Configuration: $INSTALL_PREFIX/config/
  ✅ Systemd Service: dsmil-monitor.service
  ✅ Udev Rules: Automatic device permissions

NEXT STEPS:

1. VERIFY INSTALLATION:
   sudo dmesg | grep -i dsmil
   lsmod | grep dsmil
   ls -la /dev/dsmil*

2. TEST CHUNKED IOCTL:
   cd "$SCRIPT_DIR"
   python3 test_chunked_ioctl.py --dry-run

3. START MONITORING (Optional):
$(if [[ "$INSTALL_MONITORING" == "true" ]]; then
echo "   sudo systemctl start dsmil-monitor"
echo "   $INSTALL_PREFIX/monitoring/start_monitoring_session.sh"
else
echo "   (Monitoring system not installed)"
fi)

4. RUN PHASE 2A TESTS:
   python3 test_tokens_with_module.py
   python3 test_smi_direct.py

CONFIGURATION FILES:
  Main Config: $INSTALL_PREFIX/config/dsmil.json
  Monitoring:  $INSTALL_PREFIX/config/monitoring.json
  Safety:      $INSTALL_PREFIX/config/safety.json

LOG FILES:
  Installation: $LOGS_DIR/installer.log
  System Logs:  /var/log/syslog (search for dsmil)
  Monitoring:   $INSTALL_PREFIX/monitoring/logs/

ROLLBACK:
  If needed: $INSTALL_PREFIX/bin/rollback_dsmil.sh

DOCUMENTATION:
  Phase 2A Guide: docs/PHASE2_CHUNKED_IOCTL_SOLUTION.md
  Monitoring Guide: monitoring/README.md
  API Reference: docs/API_REFERENCE.md

⚠️  IMPORTANT NOTES:
  - Logout/login required for group permissions
  - Reboot recommended for full udev integration
  - Monitor system temperature during testing
  - Use dry-run modes for initial testing

🔧 TROUBLESHOOTING:
  - Check kernel messages: sudo dmesg | tail -20
  - Verify module loading: sudo modprobe -v $DSMIL_MODULE_NAME
  - Test device access: ls -la /dev/dsmil*
  - Emergency stop: $INSTALL_PREFIX/monitoring/emergency_stop.sh

For support and updates, see: docs/DSMIL-DOCUMENTATION-INDEX.md

Installation completed at: $(date)
EOF
}

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

main() {
    # Initialize logging
    mkdir -p "$LOGS_DIR"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Show header
    clear
    section "DSMIL Phase 2A Expansion System Installer v$INSTALLER_VERSION"
    
    log "Starting DSMIL Phase 2A installation at $(date)"
    log "Arguments: $*"
    log "Working directory: $SCRIPT_DIR"
    
    # Show installation summary
    show_installation_summary
    
    # Run main installation
    if main_installation; then
        show_completion_message
        log "Installation completed successfully at $(date)"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            echo
            info "This was a dry run. No changes were made to your system."
            info "Run without --dry-run to perform the actual installation."
        fi
        
        exit 0
    else
        critical "Installation failed!"
        error "Check the log file for details: $LOGS_DIR/installer.log"
        log "Installation failed at $(date)"
        
        if [[ "$DRY_RUN" == "false" ]]; then
            echo
            error "To rollback partial installation, run:"
            error "  $INSTALL_PREFIX/bin/rollback_dsmil.sh"
        fi
        
        exit 1
    fi
}

# Run main function with all arguments
main "$@"