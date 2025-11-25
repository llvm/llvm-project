#!/bin/bash
# Documentation Reorganization Script - SIMPLE VERSION
# Moves all 94 markdown files from root to organized locations

PROJECT_ROOT="/home/john/LAT5150DRVMIL"
BACKUP_DIR="${PROJECT_ROOT}/documentation_backup_$(date +%Y%m%d_%H%M%S)"

# Create backup and destination directories
mkdir -p "${BACKUP_DIR}"
mkdir -p "${PROJECT_ROOT}/00-documentation/02-analysis/"{hardware,architecture,security,system}
mkdir -p "${PROJECT_ROOT}/00-documentation/01-planning/"{phase-4-deployment,agent-coordination,production}
mkdir -p "${PROJECT_ROOT}/00-documentation/04-progress/"{checkpoints,summaries,phases,reports}
mkdir -p "${PROJECT_ROOT}/00-documentation/03-ai-framework/"{coordination,strategies,scaling,testing}
mkdir -p "${PROJECT_ROOT}/00-documentation/00-indexes"
mkdir -p "${PROJECT_ROOT}/00-documentation/05-reference/"{guides,operations}
mkdir -p "${PROJECT_ROOT}/02-deployment/"{dsmil,reports,guides,thermal-guardian,monitoring}
mkdir -p "${PROJECT_ROOT}/03-security/"{procedures,audit}
mkdir -p "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs"
mkdir -p "${PROJECT_ROOT}/99-archive/"{organization,legacy-docs}

# Function to move file
move_file() {
    local src="$1"
    local dst="$2"

    if [ -f "${src}" ]; then
        cp "${src}" "${BACKUP_DIR}/" 2>/dev/null || true
        mv -f "${src}" "${dst}" && echo "✓ $(basename ${src})"
    fi
}

echo "Dell MIL-SPEC Documentation Reorganization"
echo "==========================================="
echo ""
echo "Files before: $(find ${PROJECT_ROOT} -maxdepth 1 -name '*.md' -type f | wc -l)"
echo "Backup: ${BACKUP_DIR}"
echo ""

# Analysis Files
echo "Moving Analysis Files..."
move_file "${PROJECT_ROOT}/DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-72-DEVICE-DISCOVERY-COMPLETE.md"
move_file "${PROJECT_ROOT}/DSMIL-DEVICE-FUNCTION-ANALYSIS.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-DEVICE-FUNCTION-ANALYSIS.md"
move_file "${PROJECT_ROOT}/GNA_ACCELERATION_ANALYSIS.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/GNA_ACCELERATION_ANALYSIS.md"
move_file "${PROJECT_ROOT}/FULL_DEVICE_COVERAGE_ANALYSIS.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/FULL_DEVICE_COVERAGE_ANALYSIS.md"
move_file "${PROJECT_ROOT}/DSMIL-CONTROL-MECHANISM-INVESTIGATION.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-CONTROL-MECHANISM-INVESTIGATION.md"
move_file "${PROJECT_ROOT}/DSMIL-DEBUG-INFRASTRUCTURE-COMPLETE.md" "${PROJECT_ROOT}/00-documentation/02-analysis/hardware/DSMIL-DEBUG-INFRASTRUCTURE-COMPLETE.md"
move_file "${PROJECT_ROOT}/DEVICE_ARCHITECTURE_INSIGHT.md" "${PROJECT_ROOT}/00-documentation/02-analysis/architecture/DEVICE_ARCHITECTURE_INSIGHT.md"
move_file "${PROJECT_ROOT}/NSA_DEVICE_IDENTIFICATION_FINAL.md" "${PROJECT_ROOT}/00-documentation/02-analysis/security/NSA_DEVICE_IDENTIFICATION_FINAL.md"
move_file "${PROJECT_ROOT}/NSA_HARDWARE_THREAT_ASSESSMENT.md" "${PROJECT_ROOT}/00-documentation/02-analysis/security/NSA_HARDWARE_THREAT_ASSESSMENT.md"
move_file "${PROJECT_ROOT}/SYSTEM-FREEZE-ANALYSIS.md" "${PROJECT_ROOT}/00-documentation/02-analysis/system/SYSTEM-FREEZE-ANALYSIS.md"
move_file "${PROJECT_ROOT}/DEV_RECOVERY_LOG.md" "${PROJECT_ROOT}/00-documentation/02-analysis/system/DEV_RECOVERY_LOG.md"

# Deployment Files
echo "Moving Deployment Files..."
move_file "${PROJECT_ROOT}/INSTALLER_README.md" "${PROJECT_ROOT}/02-deployment/dsmil/INSTALLER_README.md"
move_file "${PROJECT_ROOT}/DEPLOYMENT_README.md" "${PROJECT_ROOT}/02-deployment/README.md"
move_file "${PROJECT_ROOT}/DEPLOYMENT_SUMMARY.md" "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_SUMMARY.md"
move_file "${PROJECT_ROOT}/DEPLOYMENT_EXECUTION_SUMMARY.md" "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_EXECUTION_SUMMARY.md"
move_file "${PROJECT_ROOT}/DEPLOYMENT_DEBUGGING_FINAL_REPORT.md" "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_DEBUGGING_FINAL_REPORT.md"
move_file "${PROJECT_ROOT}/DEPLOYMENT_SUCCESS_PHASE2A.md" "${PROJECT_ROOT}/02-deployment/reports/DEPLOYMENT_SUCCESS_PHASE2A.md"
move_file "${PROJECT_ROOT}/MODULE-LOAD-STATUS.md" "${PROJECT_ROOT}/02-deployment/reports/MODULE-LOAD-STATUS.md"
move_file "${PROJECT_ROOT}/infrastructure-ready-report.md" "${PROJECT_ROOT}/02-deployment/reports/infrastructure-ready-report.md"
move_file "${PROJECT_ROOT}/SECURE_DEPLOYMENT_USAGE.md" "${PROJECT_ROOT}/02-deployment/guides/SECURE_DEPLOYMENT_USAGE.md"
move_file "${PROJECT_ROOT}/THERMAL_GUARDIAN_DEPLOYMENT.md" "${PROJECT_ROOT}/02-deployment/thermal-guardian/THERMAL_GUARDIAN_DEPLOYMENT.md"
move_file "${PROJECT_ROOT}/README_THERMAL_GUARDIAN.md" "${PROJECT_ROOT}/02-deployment/thermal-guardian/README_THERMAL_GUARDIAN.md"
move_file "${PROJECT_ROOT}/DSMIL_MONITORING_SETUP_COMPLETE.md" "${PROJECT_ROOT}/02-deployment/monitoring/DSMIL_MONITORING_SETUP_COMPLETE.md"
move_file "${PROJECT_ROOT}/DSMIL_READONLY_MONITOR_COMPLETE.md" "${PROJECT_ROOT}/02-deployment/monitoring/DSMIL_READONLY_MONITOR_COMPLETE.md"
move_file "${PROJECT_ROOT}/MONITORING_FRAMEWORK_COMPLETE.md" "${PROJECT_ROOT}/02-deployment/monitoring/MONITORING_FRAMEWORK_COMPLETE.md"
move_file "${PROJECT_ROOT}/TRANSPARENT_OPERATION_VALIDATION.md" "${PROJECT_ROOT}/02-deployment/monitoring/TRANSPARENT_OPERATION_VALIDATION.md"

# Security Files
echo "Moving Security Files..."
move_file "${PROJECT_ROOT}/DSMIL-SECURITY-SAFETY-MEASURES.md" "${PROJECT_ROOT}/03-security/procedures/DSMIL-SECURITY-SAFETY-MEASURES.md"
move_file "${PROJECT_ROOT}/CRITICAL_SAFETY_WARNING.md" "${PROJECT_ROOT}/03-security/procedures/CRITICAL_SAFETY_WARNING.md"
move_file "${PROJECT_ROOT}/COMPLETE_SAFETY_PROTOCOL.md" "${PROJECT_ROOT}/03-security/procedures/COMPLETE_SAFETY_PROTOCOL.md"
move_file "${PROJECT_ROOT}/emergency-recovery-procedures.md" "${PROJECT_ROOT}/03-security/procedures/emergency-recovery-procedures.md"
move_file "${PROJECT_ROOT}/infrastructure-safety-checklist.md" "${PROJECT_ROOT}/03-security/procedures/infrastructure-safety-checklist.md"
move_file "${PROJECT_ROOT}/SECURITY_FIXES_REPORT.md" "${PROJECT_ROOT}/03-security/audit/SECURITY_FIXES_REPORT.md"

# Progress Files
echo "Moving Progress Files..."
move_file "${PROJECT_ROOT}/FINAL-PROGRESS-WITH-ORGANIZATION-20250727.md" "${PROJECT_ROOT}/00-documentation/04-progress/checkpoints/FINAL-PROGRESS-WITH-ORGANIZATION-20250727.md"
move_file "${PROJECT_ROOT}/PRODUCTION_GO_LIVE_DECISION.md" "${PROJECT_ROOT}/00-documentation/04-progress/checkpoints/PRODUCTION_GO_LIVE_DECISION.md"
move_file "${PROJECT_ROOT}/COMPLETE_PROJECT_RECORD.md" "${PROJECT_ROOT}/00-documentation/04-progress/summaries/COMPLETE_PROJECT_RECORD.md"
move_file "${PROJECT_ROOT}/PROJECT_COMPLETE_SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/summaries/PROJECT_COMPLETE_SUMMARY.md"
move_file "${PROJECT_ROOT}/EXECUTIVE_SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/summaries/EXECUTIVE_SUMMARY.md"
move_file "${PROJECT_ROOT}/INTEGRATION_SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/summaries/INTEGRATION_SUMMARY.md"
move_file "${PROJECT_ROOT}/TACTICAL-EXECUTION-SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/summaries/TACTICAL-EXECUTION-SUMMARY.md"
move_file "${PROJECT_ROOT}/PHASE-1-DISCOVERY-ANALYSIS-REPORT.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE-1-DISCOVERY-ANALYSIS-REPORT.md"
move_file "${PROJECT_ROOT}/PHASE-2-FOUNDATION-PROGRESS.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE-2-FOUNDATION-PROGRESS.md"
move_file "${PROJECT_ROOT}/PHASE1_TESTING_COMPLETE_REPORT.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE1_TESTING_COMPLETE_REPORT.md"
move_file "${PROJECT_ROOT}/PHASE2_NEXT_STEPS.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE2_NEXT_STEPS.md"
move_file "${PROJECT_ROOT}/PHASE3-TOKEN-TESTING-SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE3-TOKEN-TESTING-SUMMARY.md"
move_file "${PROJECT_ROOT}/PHASE_2_COMPLETION_SUMMARY.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_COMPLETION_SUMMARY.md"
move_file "${PROJECT_ROOT}/PHASE_2_COMPREHENSIVE_ENHANCEMENT_PLAN.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_COMPREHENSIVE_ENHANCEMENT_PLAN.md"
move_file "${PROJECT_ROOT}/PHASE_2_TPM_ENHANCED_PLAN.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_2_TPM_ENHANCED_PLAN.md"
move_file "${PROJECT_ROOT}/PHASE_3_INTEGRATION_COMPLETE.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE_3_INTEGRATION_COMPLETE.md"
move_file "${PROJECT_ROOT}/PHASE2A_TACTICAL_ORCHESTRATION_PLAN.md" "${PROJECT_ROOT}/00-documentation/04-progress/phases/PHASE2A_TACTICAL_ORCHESTRATION_PLAN.md"
move_file "${PROJECT_ROOT}/READY_FOR_TESTING.md" "${PROJECT_ROOT}/00-documentation/04-progress/reports/READY_FOR_TESTING.md"

# Planning Files
echo "Moving Planning Files..."
move_file "${PROJECT_ROOT}/DSMIL-PRODUCTION-TIMELINE.md" "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/DSMIL-PRODUCTION-TIMELINE.md"
move_file "${PROJECT_ROOT}/STRATEGIC_PATH_FORWARD.md" "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/STRATEGIC_PATH_FORWARD.md"
move_file "${PROJECT_ROOT}/UNIFIED-DSMIL-CONTROL-STRATEGY.md" "${PROJECT_ROOT}/00-documentation/01-planning/phase-4-deployment/UNIFIED-DSMIL-CONTROL-STRATEGY.md"
move_file "${PROJECT_ROOT}/DSMIL-AGENT-COORDINATION-PLAN.md" "${PROJECT_ROOT}/00-documentation/01-planning/agent-coordination/DSMIL-AGENT-COORDINATION-PLAN.md"
move_file "${PROJECT_ROOT}/PRODUCTION-DSMIL-AGENT-TEAM-PLAN.md" "${PROJECT_ROOT}/00-documentation/01-planning/agent-coordination/PRODUCTION-DSMIL-AGENT-TEAM-PLAN.md"
move_file "${PROJECT_ROOT}/PRODUCTION_DEPLOYMENT_EXECUTIVE_SUMMARY.md" "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_DEPLOYMENT_EXECUTIVE_SUMMARY.md"
move_file "${PROJECT_ROOT}/PRODUCTION_INTERFACE_PLAN.md" "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_INTERFACE_PLAN.md"
move_file "${PROJECT_ROOT}/PRODUCTION_UPDATE_POWER_MANAGEMENT.md" "${PROJECT_ROOT}/00-documentation/01-planning/production/PRODUCTION_UPDATE_POWER_MANAGEMENT.md"
move_file "${PROJECT_ROOT}/DEBIAN-COMPATIBILITY-NOTE.md" "${PROJECT_ROOT}/00-documentation/01-planning/production/DEBIAN-COMPATIBILITY-NOTE.md"

# AI Framework Files
echo "Moving AI Framework Files..."
move_file "${PROJECT_ROOT}/AGENT_COMMUNICATION_PROTOCOLS.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/coordination/AGENT_COMMUNICATION_PROTOCOLS.md"
move_file "${PROJECT_ROOT}/AGENT_TEAM_COORDINATION_ACTIVATED.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/coordination/AGENT_TEAM_COORDINATION_ACTIVATED.md"
move_file "${PROJECT_ROOT}/AI-AGENT-NAVIGATION.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/strategies/AI-AGENT-NAVIGATION.md"
move_file "${PROJECT_ROOT}/ASYNC-DEVELOPMENT-MAP.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/strategies/ASYNC-DEVELOPMENT-MAP.md"
move_file "${PROJECT_ROOT}/500-AGENT-SCALING-ANALYSIS.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/500-AGENT-SCALING-ANALYSIS.md"
move_file "${PROJECT_ROOT}/500-AGENT-TASK-DIVISION.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/500-AGENT-TASK-DIVISION.md"
move_file "${PROJECT_ROOT}/SCALED-AGENT-TASK-DIVISION.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/scaling/SCALED-AGENT-TASK-DIVISION.md"
move_file "${PROJECT_ROOT}/test_cross_project_learning.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/testing/test_cross_project_learning.md"
move_file "${PROJECT_ROOT}/test_cross_project_learning_2.md" "${PROJECT_ROOT}/00-documentation/03-ai-framework/testing/test_cross_project_learning_2.md"

# Navigation/Index Files
echo "Moving Navigation Files..."
move_file "${PROJECT_ROOT}/MASTER-NAVIGATION.md" "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER-NAVIGATION.md"
move_file "${PROJECT_ROOT}/EXECUTION-FLOW.md" "${PROJECT_ROOT}/00-documentation/00-indexes/EXECUTION-FLOW.md"
move_file "${PROJECT_ROOT}/DIRECTORY-INDEX.md" "${PROJECT_ROOT}/00-documentation/00-indexes/DIRECTORY-INDEX.md"
move_file "${PROJECT_ROOT}/MASTER_DOCUMENTATION_INDEX.md" "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER_DOCUMENTATION_INDEX.md"
move_file "${PROJECT_ROOT}/MASTER_EXECUTION_RECORD.md" "${PROJECT_ROOT}/00-documentation/00-indexes/MASTER_EXECUTION_RECORD.md"
move_file "${PROJECT_ROOT}/PLANNING-COMPLETENESS-MATRIX.md" "${PROJECT_ROOT}/00-documentation/00-indexes/PLANNING-COMPLETENESS-MATRIX.md"
move_file "${PROJECT_ROOT}/PROJECT-ARCHITECTURE-FLOWCHART.md" "${PROJECT_ROOT}/00-documentation/00-indexes/PROJECT-ARCHITECTURE-FLOWCHART.md"
move_file "${PROJECT_ROOT}/DOCUMENTATION-CRAWL-RESULTS.md" "${PROJECT_ROOT}/00-documentation/00-indexes/DOCUMENTATION-CRAWL-RESULTS.md"
move_file "${PROJECT_ROOT}/ORGANIZATION-COMPLETE.md" "${PROJECT_ROOT}/00-documentation/00-indexes/ORGANIZATION-COMPLETE.md"
move_file "${PROJECT_ROOT}/ORGANIZATION_UPDATE.md" "${PROJECT_ROOT}/00-documentation/00-indexes/ORGANIZATION_UPDATE.md"

# TPM2 Files
echo "Moving TPM2 Files..."
move_file "${PROJECT_ROOT}/TPM2_COMPATIBILITY_IMPLEMENTATION_SUMMARY.md" "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_COMPATIBILITY_IMPLEMENTATION_SUMMARY.md"
move_file "${PROJECT_ROOT}/TPM2_OPERATIONAL_PROCEDURES.md" "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_OPERATIONAL_PROCEDURES.md"
move_file "${PROJECT_ROOT}/TPM2_PRODUCTION_DEPLOYMENT_REPORT.md" "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_PRODUCTION_DEPLOYMENT_REPORT.md"
move_file "${PROJECT_ROOT}/TPM2_PRODUCTION_DEPLOYMENT_STATUS.md" "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TPM2_PRODUCTION_DEPLOYMENT_STATUS.md"
move_file "${PROJECT_ROOT}/TRACK_C_IMPLEMENTATION_COMPLETE.md" "${PROJECT_ROOT}/tpm2_compat/c_acceleration/package_docs/TRACK_C_IMPLEMENTATION_COMPLETE.md"

# Reference Files
echo "Moving Reference Files..."
move_file "${PROJECT_ROOT}/MILITARY_TOKEN_ACTIVATION_COMPLETE.md" "${PROJECT_ROOT}/00-documentation/05-reference/guides/MILITARY_TOKEN_ACTIVATION_COMPLETE.md"
move_file "${PROJECT_ROOT}/MILITARY_TOKEN_ACTIVATION_GUIDE.md" "${PROJECT_ROOT}/00-documentation/05-reference/guides/MILITARY_TOKEN_ACTIVATION_GUIDE.md"
move_file "${PROJECT_ROOT}/TESTING_USAGE_INSTRUCTIONS.md" "${PROJECT_ROOT}/00-documentation/05-reference/guides/TESTING_USAGE_INSTRUCTIONS.md"
move_file "${PROJECT_ROOT}/TOKEN_CORRELATION_USAGE.md" "${PROJECT_ROOT}/00-documentation/05-reference/guides/TOKEN_CORRELATION_USAGE.md"
move_file "${PROJECT_ROOT}/QUICK_REFERENCE_OPERATIONS_GUIDE.md" "${PROJECT_ROOT}/00-documentation/05-reference/operations/QUICK_REFERENCE_OPERATIONS_GUIDE.md"

# Archive Files
echo "Moving Archive Files..."
move_file "${PROJECT_ROOT}/REORGANIZATION-COMPLETE.md" "${PROJECT_ROOT}/99-archive/organization/REORGANIZATION-COMPLETE.md"
move_file "${PROJECT_ROOT}/REORGANIZATION-PLAN.md" "${PROJECT_ROOT}/99-archive/organization/REORGANIZATION-PLAN.md"
move_file "${PROJECT_ROOT}/ORGANIZED_PROJECT_ARCHIVE.md" "${PROJECT_ROOT}/99-archive/organization/ORGANIZED_PROJECT_ARCHIVE.md"
move_file "${PROJECT_ROOT}/CHANGELOG.md" "${PROJECT_ROOT}/99-archive/legacy-docs/CHANGELOG.md"

echo ""
echo "==========================================="
echo "Reorganization Complete!"
echo "Files after: $(find ${PROJECT_ROOT} -maxdepth 1 -name '*.md' -type f | wc -l)"
echo "Backup location: ${BACKUP_DIR}"
echo ""
echo "Remaining files in root:"
find "${PROJECT_ROOT}" -maxdepth 1 -name "*.md" -type f -exec basename {} \;
