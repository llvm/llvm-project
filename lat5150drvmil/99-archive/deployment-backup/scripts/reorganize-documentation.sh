#!/bin/bash
# Documentation Reorganization Script - Dell MIL-SPEC Platform
# Based on DOCGEN Agent analysis: 94+ markdown files need organization
#
# Usage:
#   ./reorganize-documentation.sh --dry-run  # Preview changes
#   ./reorganize-documentation.sh            # Execute with backup

set -euo pipefail

PROJECT_ROOT="/home/john/LAT5150DRVMIL"
BACKUP_DIR="${PROJECT_ROOT}/documentation_backup_$(date +%Y%m%d_%H%M%S)"
DRY_RUN=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            cat << EOF
Documentation Reorganization Script

Usage: $0 [OPTIONS]

Options:
    --dry-run    Preview changes without moving files
    --verbose    Show detailed output
    --help       Show this help message

This script reorganizes 60+ misplaced documentation files based on
the canonical DIRECTORY-STRUCTURE.md layout.

Categories:
  - Analysis files → 00-documentation/02-analysis/
  - Deployment files → 02-deployment/
  - Security files → 03-security/
  - Progress files → 00-documentation/04-progress/
  - TPM2 user guides → tpm2_compat/c_acceleration/package_docs/

All files are backed up to: documentation_backup_TIMESTAMP/
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to log messages
log() {
    local level=$1
    shift
    local message="$@"

    case $level in
        INFO)
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        WARNING)
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        ERROR)
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to safely move file
safe_move() {
    local src="$1"
    local dst="$2"
    local category="$3"

    # Check if source exists
    if [[ ! -f "${src}" ]]; then
        [[ "$VERBOSE" == "true" ]] && log WARNING "Source not found: ${src}"
        return 0
    fi

    # Create destination directory
    local dst_dir=$(dirname "${dst}")

    if [[ "${DRY_RUN}" == "true" ]]; then
        echo "  [DRY-RUN] Would move:"
        echo "    FROM: ${src}"
        echo "    TO:   ${dst}"
        return 0
    fi

    # Create backup if not already backed up
    local basename=$(basename "${src}")
    if [[ ! -f "${BACKUP_DIR}/${basename}" ]]; then
        mkdir -p "${BACKUP_DIR}"
        cp "${src}" "${BACKUP_DIR}/" 2>/dev/null || true
    fi

    # Create destination directory
    mkdir -p "${dst_dir}"

    # Move file
    if mv "${src}" "${dst}"; then
        log SUCCESS "Moved $(basename ${src}) → ${dst_dir}/"
        return 0
    else
        log ERROR "Failed to move ${src}"
        return 1
    fi
}

# Banner
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Dell MIL-SPEC Documentation Reorganization"
echo "═══════════════════════════════════════════════════════"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    log WARNING "DRY RUN MODE - No files will be moved"
    echo ""
fi

# Create backup directory
if [[ "${DRY_RUN}" == "false" ]]; then
    mkdir -p "${BACKUP_DIR}"
    log INFO "Backup directory: ${BACKUP_DIR}"
    echo ""
fi

# Track statistics
TOTAL_MOVES=0
SUCCESSFUL_MOVES=0
FAILED_MOVES=0

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 1: Analysis Files → 00-documentation/02-analysis/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 1: Analysis Files ==="
((TOTAL_MOVES+=8))

safe_move \
    "${PROJECT_ROOT}/DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL-12-DEVICE-UPDATE-COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-12-DEVICE-UPDATE-COMPLETE.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL-DEVICE-FUNCTION-ANALYSIS.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-DEVICE-FUNCTION-ANALYSIS.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/GNA_ACCELERATION_ANALYSIS.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/GNA_ACCELERATION_ANALYSIS.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/FULL_DEVICE_COVERAGE_ANALYSIS.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/FULL_DEVICE_COVERAGE_ANALYSIS.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEVICE_ARCHITECTURE_INSIGHT.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/architecture/DEVICE_ARCHITECTURE_INSIGHT.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/NSA_DEVICE_IDENTIFICATION_FINAL.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/security/NSA_DEVICE_IDENTIFICATION_FINAL.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/NSA_HARDWARE_THREAT_ASSESSMENT.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/security/NSA_HARDWARE_THREAT_ASSESSMENT.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 2: Deployment Files → 02-deployment/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 2: Deployment Files ==="
((TOTAL_MOVES+=9))

# Create deployment structure
mkdir -p "${PROJECT_ROOT}/02-deployment/"{dsmil,reports,guides,thermal-guardian,monitoring}

safe_move \
    "${PROJECT_ROOT}/INSTALLER_README.md" \
    "${PROJECT_ROOT}/02-deployment/dsmil/INSTALLER_README.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEPLOYMENT_README.md" \
    "${PROJECT_ROOT}/02-deployment/README.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEPLOYMENT_SUMMARY.md" \
    "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_SUMMARY.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEPLOYMENT_EXECUTION_SUMMARY.md" \
    "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_EXECUTION_SUMMARY.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEPLOYMENT_DEBUGGING_FINAL_REPORT.md" \
    "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_DEBUGGING_FINAL_REPORT.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEPLOYMENT_SUCCESS_PHASE2A.md" \
    "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_SUCCESS_PHASE2A.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/SECURE_DEPLOYMENT_USAGE.md" \
    "${PROJECT_ROOT}/02-deployment/guides/SECURE_DEPLOYMENT_USAGE.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/THERMAL_GUARDIAN_DEPLOYMENT.md" \
    "${PROJECT_ROOT}/02-deployment/thermal-guardian/THERMAL_GUARDIAN_DEPLOYMENT.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL_MONITORING_SETUP_COMPLETE.md" \
    "${PROJECT_ROOT}/02-deployment/monitoring/DSMIL_MONITORING_SETUP_COMPLETE.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 3: Security Files → 03-security/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 3: Security Files ==="
((TOTAL_MOVES+=6))

# Create security structure
mkdir -p "${PROJECT_ROOT}/03-security/"{procedures,audit}

safe_move \
    "${PROJECT_ROOT}/DSMIL-SECURITY-SAFETY-MEASURES.md" \
    "${PROJECT_ROOT}/03-security/procedures/DSMIL-SECURITY-SAFETY-MEASURES.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/CRITICAL_SAFETY_WARNING.md" \
    "${PROJECT_ROOT}/03-security/procedures/CRITICAL_SAFETY_WARNING.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/COMPLETE_SAFETY_PROTOCOL.md" \
    "${PROJECT_ROOT}/03-security/procedures/COMPLETE_SAFETY_PROTOCOL.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/emergency-recovery-procedures.md" \
    "${PROJECT_ROOT}/03-security/procedures/emergency-recovery-procedures.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/infrastructure-safety-checklist.md" \
    "${PROJECT_ROOT}/03-security/procedures/infrastructure-safety-checklist.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/SECURITY_FIXES_REPORT.md" \
    "${PROJECT_ROOT}/03-security/audit/SECURITY_FIXES_REPORT.md" \
    "Security" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# Summary
#──────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════"
echo "  Reorganization Summary"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Total relocations planned: ${TOTAL_MOVES}"
echo "Successful moves: ${SUCCESSFUL_MOVES}"
echo "Failed moves: ${FAILED_MOVES}"
echo ""

if [[ "${DRY_RUN}" == "false" ]]; then
    echo "Backup location: ${BACKUP_DIR}"
    echo ""
    echo "To restore from backup:"
    echo "  cp ${BACKUP_DIR}/* ${PROJECT_ROOT}/"
    echo ""
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "This was a DRY RUN. No files were moved."
    echo "Run without --dry-run to execute changes."
else
    log SUCCESS "Documentation reorganization complete!"
fi

echo ""

exit 0
