#!/bin/bash
#
# Dell MIL-SPEC Platform - Manual Installation Detection
# Detects and reports manual installation artifacts
#
# Version: 1.0.0
# Author: Claude Agent Framework - DEPLOYER Agent
# Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
#
# Exit codes:
#   0 - Manual installation found
#   1 - Clean system (no manual installation)
#   2 - Partial installation detected
#

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

SCRIPT_VERSION="1.0.0"
OUTPUT_FORMAT="json"  # json, text, summary
VERBOSE=false

# Manual installation indicators
MANUAL_MODULE_DSMIL_PATTERN="/lib/modules/*/extra/dsmil-72dev.ko"
MANUAL_MODULE_TPM2_PATTERN="/lib/modules/*/kernel/drivers/tpm/tpm2_accel_early.ko"
MANUAL_INSTALL_DIR="/opt/dsmil"
MANUAL_SERVICE="/etc/systemd/system/dsmil-monitor.service"
MANUAL_CONFIGS=(
    "/etc/modprobe.d/dsmil-72dev.conf"
    "/etc/modules-load.d/tpm2-acceleration.conf"
    "/etc/modprobe.d/tpm2-acceleration.conf"
)
MANUAL_TOOLS=(
    "/usr/local/bin/dsmil-status"
    "/usr/local/bin/dsmil-monitor"
)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Detection results
FOUND_ARTIFACTS=()
FOUND_MODULES=()
FOUND_CONFIGS=()
FOUND_SERVICES=()
FOUND_DIRECTORIES=()
FOUND_TOOLS=()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_verbose() {
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[VERBOSE]${NC} $*" >&2
    fi
}

check_file() {
    local file="$1"
    if [[ -f "$file" ]]; then
        log_verbose "Found file: $file"
        return 0
    fi
    return 1
}

check_directory() {
    local dir="$1"
    if [[ -d "$dir" ]]; then
        log_verbose "Found directory: $dir"
        return 0
    fi
    return 1
}

check_module_loaded() {
    local module="$1"
    if lsmod | grep -q "^${module}"; then
        log_verbose "Module loaded: $module"
        return 0
    fi
    return 1
}

check_service() {
    local service="$1"
    if systemctl list-unit-files | grep -q "$service"; then
        log_verbose "Service found: $service"
        return 0
    fi
    return 1
}

# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

detect_kernel_modules() {
    log_verbose "Detecting kernel modules..."

    # Check for DSMIL module files
    local dsmil_modules=($(find /lib/modules -name "dsmil-72dev.ko" 2>/dev/null || true))
    for module in "${dsmil_modules[@]}"; do
        FOUND_MODULES+=("$module")
        FOUND_ARTIFACTS+=("module:dsmil-72dev:$module")
    done

    # Check for TPM2 module files
    local tpm2_modules=($(find /lib/modules -name "tpm2_accel_early.ko" 2>/dev/null || true))
    for module in "${tpm2_modules[@]}"; do
        FOUND_MODULES+=("$module")
        FOUND_ARTIFACTS+=("module:tpm2_accel_early:$module")
    done

    # Check if modules are loaded
    if check_module_loaded "dsmil_72dev" || check_module_loaded "dsmil"; then
        FOUND_ARTIFACTS+=("module_loaded:dsmil")
    fi

    if check_module_loaded "tpm2_accel_early"; then
        FOUND_ARTIFACTS+=("module_loaded:tpm2_accel_early")
    fi

    log_verbose "Found ${#FOUND_MODULES[@]} kernel module files"
}

detect_installation_directory() {
    log_verbose "Detecting installation directory..."

    if check_directory "$MANUAL_INSTALL_DIR"; then
        FOUND_DIRECTORIES+=("$MANUAL_INSTALL_DIR")
        FOUND_ARTIFACTS+=("directory:install:$MANUAL_INSTALL_DIR")

        # Check subdirectories
        local subdirs=("monitoring" "config" "logs" "bin")
        for subdir in "${subdirs[@]}"; do
            local path="${MANUAL_INSTALL_DIR}/${subdir}"
            if check_directory "$path"; then
                FOUND_DIRECTORIES+=("$path")
                FOUND_ARTIFACTS+=("directory:${subdir}:${path}")
            fi
        done

        # Count files in installation
        local file_count=$(find "$MANUAL_INSTALL_DIR" -type f 2>/dev/null | wc -l)
        log_verbose "Found $file_count files in $MANUAL_INSTALL_DIR"
    fi
}

detect_systemd_services() {
    log_verbose "Detecting systemd services..."

    if check_file "$MANUAL_SERVICE"; then
        FOUND_SERVICES+=("$MANUAL_SERVICE")
        FOUND_ARTIFACTS+=("service:dsmil-monitor:$MANUAL_SERVICE")

        # Check service status
        if systemctl is-active --quiet dsmil-monitor 2>/dev/null; then
            FOUND_ARTIFACTS+=("service_active:dsmil-monitor")
        fi

        if systemctl is-enabled --quiet dsmil-monitor 2>/dev/null; then
            FOUND_ARTIFACTS+=("service_enabled:dsmil-monitor")
        fi
    fi
}

detect_configuration_files() {
    log_verbose "Detecting configuration files..."

    for config in "${MANUAL_CONFIGS[@]}"; do
        if check_file "$config"; then
            FOUND_CONFIGS+=("$config")
            FOUND_ARTIFACTS+=("config:$(basename $config):$config")
        fi
    done

    # Check for additional config files
    if check_directory "/etc/udev/rules.d"; then
        local udev_rules=$(find /etc/udev/rules.d -name "*dsmil*" 2>/dev/null || true)
        for rule in $udev_rules; do
            FOUND_CONFIGS+=("$rule")
            FOUND_ARTIFACTS+=("config:udev:$rule")
        done
    fi
}

detect_manual_tools() {
    log_verbose "Detecting manual tools..."

    for tool in "${MANUAL_TOOLS[@]}"; do
        if check_file "$tool"; then
            FOUND_TOOLS+=("$tool")
            FOUND_ARTIFACTS+=("tool:$(basename $tool):$tool")
        fi
    done

    # Check for additional tools in /usr/local/bin
    local local_tools=$(find /usr/local/bin -name "*dsmil*" -o -name "*milspec*" 2>/dev/null || true)
    for tool in $local_tools; do
        if [[ ! " ${FOUND_TOOLS[@]} " =~ " ${tool} " ]]; then
            FOUND_TOOLS+=("$tool")
            FOUND_ARTIFACTS+=("tool:$(basename $tool):$tool")
        fi
    done
}

detect_device_files() {
    log_verbose "Detecting device files..."

    # Check for DSMIL device nodes
    if [[ -e "/dev/dsmil0" ]]; then
        FOUND_ARTIFACTS+=("device:/dev/dsmil0")
    fi

    local dsmil_devices=$(find /dev -name "dsmil*" 2>/dev/null || true)
    for device in $dsmil_devices; do
        FOUND_ARTIFACTS+=("device:$device")
    done

    # Check for TPM2 acceleration device
    if [[ -e "/dev/tpm2_accel_early" ]]; then
        FOUND_ARTIFACTS+=("device:/dev/tpm2_accel_early")
    fi
}

detect_groups() {
    log_verbose "Detecting system groups..."

    if getent group dsmil >/dev/null 2>&1; then
        FOUND_ARTIFACTS+=("group:dsmil")

        # Get group members
        local members=$(getent group dsmil | cut -d: -f4)
        if [[ -n "$members" ]]; then
            FOUND_ARTIFACTS+=("group_members:dsmil:$members")
        fi
    fi
}

# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

output_json() {
    local status="$1"
    local exit_code="$2"

    cat << EOF
{
  "detection_version": "$SCRIPT_VERSION",
  "timestamp": "$(date -Iseconds)",
  "hostname": "$(hostname)",
  "kernel": "$(uname -r)",
  "status": "$status",
  "exit_code": $exit_code,
  "summary": {
    "total_artifacts": ${#FOUND_ARTIFACTS[@]},
    "modules": ${#FOUND_MODULES[@]},
    "configs": ${#FOUND_CONFIGS[@]},
    "services": ${#FOUND_SERVICES[@]},
    "directories": ${#FOUND_DIRECTORIES[@]},
    "tools": ${#FOUND_TOOLS[@]}
  },
  "artifacts": [
EOF

    local first=true
    for artifact in "${FOUND_ARTIFACTS[@]}"; do
        if [[ "$first" == "false" ]]; then
            echo ","
        fi
        first=false

        local type=$(echo "$artifact" | cut -d: -f1)
        local name=$(echo "$artifact" | cut -d: -f2)
        local path=$(echo "$artifact" | cut -d: -f3-)

        echo -n "    {\"type\": \"$type\", \"name\": \"$name\""
        if [[ -n "$path" && "$path" != "$name" ]]; then
            echo -n ", \"path\": \"$path\""
        fi
        echo -n "}"
    done

    echo ""
    echo "  ],"

    # Detailed sections
    echo "  \"modules\": ["
    first=true
    for module in "${FOUND_MODULES[@]}"; do
        if [[ "$first" == "false" ]]; then echo ","; fi
        first=false
        echo -n "    \"$module\""
    done
    echo ""
    echo "  ],"

    echo "  \"configurations\": ["
    first=true
    for config in "${FOUND_CONFIGS[@]}"; do
        if [[ "$first" == "false" ]]; then echo ","; fi
        first=false
        echo -n "    \"$config\""
    done
    echo ""
    echo "  ],"

    echo "  \"services\": ["
    first=true
    for service in "${FOUND_SERVICES[@]}"; do
        if [[ "$first" == "false" ]]; then echo ","; fi
        first=false
        echo -n "    \"$service\""
    done
    echo ""
    echo "  ],"

    echo "  \"directories\": ["
    first=true
    for directory in "${FOUND_DIRECTORIES[@]}"; do
        if [[ "$first" == "false" ]]; then echo ","; fi
        first=false
        echo -n "    \"$directory\""
    done
    echo ""
    echo "  ],"

    echo "  \"tools\": ["
    first=true
    for tool in "${FOUND_TOOLS[@]}"; do
        if [[ "$first" == "false" ]]; then echo ","; fi
        first=false
        echo -n "    \"$tool\""
    done
    echo ""
    echo "  ]"

    echo "}"
}

output_text() {
    local status="$1"

    echo "Manual Installation Detection Report"
    echo "===================================="
    echo ""
    echo "Timestamp: $(date)"
    echo "Hostname: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "Status: $status"
    echo ""
    echo "Summary:"
    echo "--------"
    echo "Total Artifacts: ${#FOUND_ARTIFACTS[@]}"
    echo "  Modules: ${#FOUND_MODULES[@]}"
    echo "  Configs: ${#FOUND_CONFIGS[@]}"
    echo "  Services: ${#FOUND_SERVICES[@]}"
    echo "  Directories: ${#FOUND_DIRECTORIES[@]}"
    echo "  Tools: ${#FOUND_TOOLS[@]}"
    echo ""

    if [[ ${#FOUND_MODULES[@]} -gt 0 ]]; then
        echo "Kernel Modules:"
        for module in "${FOUND_MODULES[@]}"; do
            echo "  - $module"
        done
        echo ""
    fi

    if [[ ${#FOUND_CONFIGS[@]} -gt 0 ]]; then
        echo "Configuration Files:"
        for config in "${FOUND_CONFIGS[@]}"; do
            echo "  - $config"
        done
        echo ""
    fi

    if [[ ${#FOUND_SERVICES[@]} -gt 0 ]]; then
        echo "Systemd Services:"
        for service in "${FOUND_SERVICES[@]}"; do
            echo "  - $service"
        done
        echo ""
    fi

    if [[ ${#FOUND_DIRECTORIES[@]} -gt 0 ]]; then
        echo "Directories:"
        for directory in "${FOUND_DIRECTORIES[@]}"; do
            echo "  - $directory"
        done
        echo ""
    fi

    if [[ ${#FOUND_TOOLS[@]} -gt 0 ]]; then
        echo "Manual Tools:"
        for tool in "${FOUND_TOOLS[@]}"; do
            echo "  - $tool"
        done
        echo ""
    fi
}

output_summary() {
    local status="$1"

    if [[ "$status" == "manual_found" ]]; then
        echo -e "${YELLOW}Manual installation detected${NC} (${#FOUND_ARTIFACTS[@]} artifacts)"
    elif [[ "$status" == "clean" ]]; then
        echo -e "${GREEN}Clean system${NC} (no manual installation)"
    elif [[ "$status" == "partial" ]]; then
        echo -e "${YELLOW}Partial installation${NC} (${#FOUND_ARTIFACTS[@]} artifacts)"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
Dell MIL-SPEC Platform - Manual Installation Detection
Version: $SCRIPT_VERSION

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -f, --format FORMAT Output format: json, text, summary (default: json)
    -v, --verbose       Verbose output to stderr

DESCRIPTION:
    Detects manual installation artifacts on the system.

EXIT CODES:
    0 - Manual installation found
    1 - Clean system (no manual installation)
    2 - Partial installation detected

EXAMPLES:
    $0                      # JSON output
    $0 --format text        # Human-readable output
    $0 --format summary     # One-line summary

OUTPUT:
    Detection results are written to stdout in the specified format.

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                echo "Unknown option: $1" >&2
                echo "Use --help for usage information" >&2
                exit 1
                ;;
        esac
    done
}

main() {
    # Parse arguments
    parse_arguments "$@"

    log_verbose "Starting manual installation detection..."

    # Run all detection functions
    detect_kernel_modules
    detect_installation_directory
    detect_systemd_services
    detect_configuration_files
    detect_manual_tools
    detect_device_files
    detect_groups

    # Determine status
    local status
    local exit_code

    if [[ ${#FOUND_ARTIFACTS[@]} -eq 0 ]]; then
        status="clean"
        exit_code=1
    elif [[ ${#FOUND_MODULES[@]} -gt 0 ]] || [[ -d "$MANUAL_INSTALL_DIR" ]]; then
        status="manual_found"
        exit_code=0
    else
        status="partial"
        exit_code=2
    fi

    log_verbose "Detection complete: $status (${#FOUND_ARTIFACTS[@]} artifacts)"

    # Output results
    case "$OUTPUT_FORMAT" in
        json)
            output_json "$status" "$exit_code"
            ;;
        text)
            output_text "$status"
            ;;
        summary)
            output_summary "$status"
            ;;
        *)
            echo "Invalid output format: $OUTPUT_FORMAT" >&2
            exit 1
            ;;
    esac

    exit $exit_code
}

# Run main
main "$@"
