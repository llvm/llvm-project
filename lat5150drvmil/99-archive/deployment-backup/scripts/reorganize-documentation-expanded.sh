#!/bin/bash
# Documentation Reorganization Script - Dell MIL-SPEC Platform (EXPANDED)
# Based on DOCGEN Agent analysis: 94+ markdown files need organization
# This version handles ALL 94 files, not just 23
#
# Usage:
#   ./reorganize-documentation-expanded.sh --dry-run  # Preview changes
#   ./reorganize-documentation-expanded.sh            # Execute with backup

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
Documentation Reorganization Script (EXPANDED - ALL 94 FILES)

Usage: $0 [OPTIONS]

Options:
    --dry-run    Preview changes without moving files
    --verbose    Show detailed output
    --help       Show this help message

This script reorganizes ALL 94 misplaced documentation files based on
the canonical DIRECTORY-STRUCTURE.md layout.

Categories:
  - Analysis files (12) → 00-documentation/02-analysis/
  - Deployment files (12) → 02-deployment/
  - Security files (6) → 03-security/
  - Progress files (18) → 00-documentation/04-progress/
  - Planning files (9) → 00-documentation/01-planning/
  - AI Framework (10) → 00-documentation/03-ai-framework/
  - Navigation/Index (10) → 00-documentation/00-indexes/
  - TPM2/Monitoring (10) → tpm2_compat/c_acceleration/package_docs/
  - Reference files (6) → 00-documentation/05-reference/
  - Archive (4) → 99-archive/

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

    # Move file (use -f to force overwrite if destination exists)
    if mv -f "${src}" "${dst}"; then
        log SUCCESS "Moved $(basename ${src}) → ${dst_dir}/"
        return 0
    else
        log ERROR "Failed to move ${src}"
        return 1
    fi
}

# Banner
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Dell MIL-SPEC Documentation Reorganization (EXPANDED)"
echo "  Processing ALL 94 markdown files"
echo "═══════════════════════════════════════════════════════════"
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

# Count files before
FILES_BEFORE=$(find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f | wc -l)
log INFO "Markdown files in root before: ${FILES_BEFORE}"
echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 1: Analysis Files → 00-documentation/02-analysis/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 1: Analysis Files (12 files) ==="
((TOTAL_MOVES+=12))

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
    "${PROJECT_ROOT}/DSMIL-CONTROL-MECHANISM-INVESTIGATION.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-CONTROL-MECHANISM-INVESTIGATION.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL-DEBUG-INFRASTRUCTURE-COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-DEBUG-INFRASTRUCTURE-COMPLETE.md" \
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

safe_move \
    "${PROJECT_ROOT}/SYSTEM-FREEZE-ANALYSIS.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/system/SYSTEM-FREEZE-ANALYSIS.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEV_RECOVERY_LOG.md" \
    "${PROJECT_ROOT}/00-documentation/02-analysis/system/DEV_RECOVERY_LOG.md" \
    "Analysis" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 2: Deployment Files → 02-deployment/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 2: Deployment Files (12 files) ==="
((TOTAL_MOVES+=12))

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
    "${PROJECT_ROOT}/MODULE-LOAD-STATUS.md" \
    "${PROJECT_ROOT}/02-deployment/reports/MODULE-LOAD-STATUS.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/infrastructure-ready-report.md" \
    "${PROJECT_ROOT}/02-deployment/reports/infrastructure-ready-report.md" \
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
    "${PROJECT_ROOT}/README_THERMAL_GUARDIAN.md" \
    "${PROJECT_ROOT}/02-deployment/thermal-guardian/README_THERMAL_GUARDIAN.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL_MONITORING_SETUP_COMPLETE.md" \
    "${PROJECT_ROOT}/02-deployment/monitoring/DSMIL_MONITORING_SETUP_COMPLETE.md" \
    "Deployment" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 3: Security Files → 03-security/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 3: Security Files (6 files) ==="
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
# CATEGORY 4: Progress & Status Files → 00-documentation/04-progress/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 4: Progress & Status Files (18 files) ==="
((TOTAL_MOVES+=18))

mkdir -p "${PROJECT_ROOT}/00-documentation/04-progress/"{checkpoints,summaries,reports,phases}

safe_move \
    "${PROJECT_ROOT}/FINAL-PROGRESS-WITH-ORGANIZATION-20250727.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/checkpoints/FINAL-PROGRESS-WITH-ORGANIZATION-20250727.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PRODUCTION_GO_LIVE_DECISION.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/checkpoints/PRODUCTION_GO_LIVE_DECISION.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/COMPLETE_PROJECT_RECORD.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/summaries/COMPLETE_PROJECT_RECORD.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PROJECT_COMPLETE_SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/summaries/PROJECT_COMPLETE_SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/EXECUTIVE_SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/summaries/EXECUTIVE_SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/INTEGRATION_SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/summaries/INTEGRATION_SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TACTICAL-EXECUTION-SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/summaries/TACTICAL-EXECUTION-SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE-1-DISCOVERY-ANALYSIS-REPORT.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE-1-DISCOVERY-ANALYSIS-REPORT.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE-2-FOUNDATION-PROGRESS.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE-2-FOUNDATION-PROGRESS.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE1_TESTING_COMPLETE_REPORT.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE1_TESTING_COMPLETE_REPORT.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE2_NEXT_STEPS.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE2_NEXT_STEPS.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE3-TOKEN-TESTING-SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE3-TOKEN-TESTING-SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE_2_COMPLETION_SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_COMPLETION_SUMMARY.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE_2_COMPREHENSIVE_ENHANCEMENT_PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_COMPREHENSIVE_ENHANCEMENT_PLAN.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE_2_TPM_ENHANCED_PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_TPM_ENHANCED_PLAN.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE_3_INTEGRATION_COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_3_INTEGRATION_COMPLETE.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PHASE2A_TACTICAL_ORCHESTRATION_PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE2A_TACTICAL_ORCHESTRATION_PLAN.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/READY_FOR_TESTING.md" \
    "${PROJECT_ROOT}/00-documentation/04-progress/reports/READY_FOR_TESTING.md" \
    "Progress" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 5: Planning & Strategy Files → 00-documentation/01-planning/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 5: Planning & Strategy Files (9 files) ==="
((TOTAL_MOVES+=9))

mkdir -p "${PROJECT_ROOT}/00-documentation/01-planning/"{phase-4-deployment,agent-coordination,production}

safe_move \
    "${PROJECT_ROOT}/DSMIL-PRODUCTION-TIMELINE.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/DSMIL-PRODUCTION-TIMELINE.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/STRATEGIC_PATH_FORWARD.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/STRATEGIC_PATH_FORWARD.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/UNIFIED-DSMIL-CONTROL-STRATEGY.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/UNIFIED-DSMIL-CONTROL-STRATEGY.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL-AGENT-COORDINATION-PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/agent-coordination/DSMIL-AGENT-COORDINATION-PLAN.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PRODUCTION-DSMIL-AGENT-TEAM-PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/agent-coordination/PRODUCTION-DSMIL-AGENT-TEAM-PLAN.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PRODUCTION_DEPLOYMENT_EXECUTIVE_SUMMARY.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_DEPLOYMENT_EXECUTIVE_SUMMARY.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PRODUCTION_INTERFACE_PLAN.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_INTERFACE_PLAN.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PRODUCTION_UPDATE_POWER_MANAGEMENT.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_UPDATE_POWER_MANAGEMENT.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DEBIAN-COMPATIBILITY-NOTE.md" \
    "${PROJECT_ROOT}/00-documentation/01-planning/production/DEBIAN-COMPATIBILITY-NOTE.md" \
    "Planning" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 6: AI Framework & Agent Coordination → 00-documentation/03-ai-framework/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 6: AI Framework & Agent Files (10 files) ==="
((TOTAL_MOVES+=10))

mkdir -p "${PROJECT_ROOT}/00-documentation/03-ai-framework/"{coordination,strategies,scaling,testing}

safe_move \
    "${PROJECT_ROOT}/AGENT_COMMUNICATION_PROTOCOLS.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/coordination/AGENT_COMMUNICATION_PROTOCOLS.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/AGENT_TEAM_COORDINATION_ACTIVATED.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/coordination/AGENT_TEAM_COORDINATION_ACTIVATED.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/AI-AGENT-NAVIGATION.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/strategies/AI-AGENT-NAVIGATION.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/ASYNC-DEVELOPMENT-MAP.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/strategies/ASYNC-DEVELOPMENT-MAP.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/500-AGENT-SCALING-ANALYSIS.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/500-AGENT-SCALING-ANALYSIS.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/500-AGENT-TASK-DIVISION.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/500-AGENT-TASK-DIVISION.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/SCALED-AGENT-TASK-DIVISION.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/SCALED-AGENT-TASK-DIVISION.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/test_cross_project_learning.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/testing/test_cross_project_learning.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/test_cross_project_learning_2.md" \
    "${PROJECT_ROOT}/00-documentation/03-ai-framework/testing/test_cross_project_learning_2.md" \
    "AI Framework" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 7: Navigation & Index Files → 00-documentation/00-indexes/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 7: Navigation & Index Files (10 files) ==="
((TOTAL_MOVES+=10))

mkdir -p "${PROJECT_ROOT}/00-documentation/00-indexes"

safe_move \
    "${PROJECT_ROOT}/MASTER-NAVIGATION.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER-NAVIGATION.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/EXECUTION-FLOW.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/EXECUTION-FLOW.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DIRECTORY-INDEX.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/DIRECTORY-INDEX.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/MASTER_DOCUMENTATION_INDEX.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER_DOCUMENTATION_INDEX.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/MASTER_EXECUTION_RECORD.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER_EXECUTION_RECORD.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PLANNING-COMPLETENESS-MATRIX.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/PLANNING-COMPLETENESS-MATRIX.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/PROJECT-ARCHITECTURE-FLOWCHART.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/PROJECT-ARCHITECTURE-FLOWCHART.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DOCUMENTATION-CRAWL-RESULTS.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/DOCUMENTATION-CRAWL-RESULTS.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/ORGANIZATION-COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/ORGANIZATION-COMPLETE.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/ORGANIZATION_UPDATE.md" \
    "${PROJECT_ROOT}/00-documentation/00-indexes/ORGANIZATION_UPDATE.md" \
    "Indexes" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 8: TPM2 & Monitoring Files → tpm2_compat/c_acceleration/package_docs/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 8: TPM2 & Monitoring Files (10 files) ==="
((TOTAL_MOVES+=10))

mkdir -p "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs"

safe_move \
    "${PROJECT_ROOT}/TPM2_COMPATIBILITY_IMPLEMENTATION_SUMMARY.md" \
    "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_COMPATIBILITY_IMPLEMENTATION_SUMMARY.md" \
    "TPM2" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TPM2_OPERATIONAL_PROCEDURES.md" \
    "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_OPERATIONAL_PROCEDURES.md" \
    "TPM2" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TPM2_PRODUCTION_DEPLOYMENT_REPORT.md" \
    "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_PRODUCTION_DEPLOYMENT_REPORT.md" \
    "TPM2" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TPM2_PRODUCTION_DEPLOYMENT_STATUS.md" \
    "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_PRODUCTION_DEPLOYMENT_STATUS.md" \
    "TPM2" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TRACK_C_IMPLEMENTATION_COMPLETE.md" \
    "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TRACK_C_IMPLEMENTATION_COMPLETE.md" \
    "TPM2" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/DSMIL_READONLY_MONITOR_COMPLETE.md" \
    "${PROJECT_ROOT}/02-deployment/monitoring/DSMIL_READONLY_MONITOR_COMPLETE.md" \
    "Monitoring" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/MONITORING_FRAMEWORK_COMPLETE.md" \
    "${PROJECT_ROOT}/02-deployment/monitoring/MONITORING_FRAMEWORK_COMPLETE.md" \
    "Monitoring" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TRANSPARENT_OPERATION_VALIDATION.md" \
    "${PROJECT_ROOT}/02-deployment/monitoring/TRANSPARENT_OPERATION_VALIDATION.md" \
    "Monitoring" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 9: Reference & Usage Files → 00-documentation/05-reference/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 9: Reference & Usage Files (6 files) ==="
((TOTAL_MOVES+=6))

mkdir -p "${PROJECT_ROOT}/00-documentation/05-reference/"{guides,operations}

safe_move \
    "${PROJECT_ROOT}/MILITARY_TOKEN_ACTIVATION_COMPLETE.md" \
    "${PROJECT_ROOT}/00-documentation/05-reference/guides/MILITARY_TOKEN_ACTIVATION_COMPLETE.md" \
    "Reference" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/MILITARY_TOKEN_ACTIVATION_GUIDE.md" \
    "${PROJECT_ROOT}/00-documentation/05-reference/guides/MILITARY_TOKEN_ACTIVATION_GUIDE.md" \
    "Reference" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TESTING_USAGE_INSTRUCTIONS.md" \
    "${PROJECT_ROOT}/00-documentation/05-reference/guides/TESTING_USAGE_INSTRUCTIONS.md" \
    "Reference" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/TOKEN_CORRELATION_USAGE.md" \
    "${PROJECT_ROOT}/00-documentation/05-reference/guides/TOKEN_CORRELATION_USAGE.md" \
    "Reference" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/QUICK_REFERENCE_OPERATIONS_GUIDE.md" \
    "${PROJECT_ROOT}/00-documentation/05-reference/operations/QUICK_REFERENCE_OPERATIONS_GUIDE.md" \
    "Reference" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# CATEGORY 10: Archive Files → 99-archive/
#──────────────────────────────────────────────────────────────────────────────
echo "═══ Category 10: Archive Files (4 files) ==="
((TOTAL_MOVES+=4))

mkdir -p "${PROJECT_ROOT}/99-archive/"{organization,legacy-docs}

safe_move \
    "${PROJECT_ROOT}/REORGANIZATION-COMPLETE.md" \
    "${PROJECT_ROOT}/99-archive/organization/REORGANIZATION-COMPLETE.md" \
    "Archive" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/REORGANIZATION-PLAN.md" \
    "${PROJECT_ROOT}/99-archive/organization/REORGANIZATION-PLAN.md" \
    "Archive" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/ORGANIZED_PROJECT_ARCHIVE.md" \
    "${PROJECT_ROOT}/99-archive/organization/ORGANIZED_PROJECT_ARCHIVE.md" \
    "Archive" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

safe_move \
    "${PROJECT_ROOT}/CHANGELOG.md" \
    "${PROJECT_ROOT}/99-archive/legacy-docs/CHANGELOG.md" \
    "Archive" && ((SUCCESSFUL_MOVES++)) || ((FAILED_MOVES++))

echo ""

#──────────────────────────────────────────────────────────────────────────────
# Summary & Validation
#──────────────────────────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════"
echo "  Reorganization Summary"
echo "═══════════════════════════════════════════════════════════"
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

    # Final validation
    FILES_AFTER=$(find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f | wc -l)
    FILES_MOVED=$((FILES_BEFORE - FILES_AFTER))

    log INFO "Markdown files in root after: ${FILES_AFTER}"
    log INFO "Files successfully moved: ${FILES_MOVED}"
    echo ""

    if [[ ${FILES_AFTER} -le 10 ]]; then
        log SUCCESS "Root directory successfully cleaned!"
        echo ""
        echo "Remaining files in root should be:"
        echo "  - README.md (project overview)"
        echo "  - Plus 5-10 essential navigation files"
    else
        log WARNING "${FILES_AFTER} markdown files still in root (expected: 5-10)"
        echo ""
        echo "Files still in root:"
        find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f -exec basename {} \;
    fi
    echo ""
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "This was a DRY RUN. No files were moved."
    echo "Run without --dry-run to execute changes."
    echo ""
else
    log SUCCESS "Documentation reorganization complete!"
    echo ""
    log INFO "Next steps:"
    echo "  1. Verify files are in correct locations"
    echo "  2. Update any hardcoded file paths"
    echo "  3. Test navigation from README.md"
    echo "  4. Consider removing backup after verification:"
    echo "     rm -rf ${BACKUP_DIR}"
fi

echo ""

exit 0
