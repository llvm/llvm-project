#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# DSMIL Unified AI Platform - Codebase Cleanup
# Version: 8.3
# Organizes and tidies up the repository structure
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  DSMIL CODEBASE CLEANUP"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

cleanup_deployment_dirs() {
    print_step "Consolidating deployment directories..."

    # We'll keep 05-deployment (newest) and archive the rest
    if [ -d "02-deployment" ]; then
        if [ ! -d "99-archive/02-deployment-backup" ]; then
            mv 02-deployment 99-archive/02-deployment-backup
            print_success "Moved 02-deployment to archive"
        fi
    fi

    if [ -d "deployment" ]; then
        if [ ! -d "99-archive/deployment-backup" ]; then
            mv deployment 99-archive/deployment-backup
            print_success "Moved deployment to archive"
        fi
    fi
}

cleanup_docs_dirs() {
    print_step "Organizing documentation..."

    # Keep 00-documentation (primary docs)
    # Archive duplicate docs directory
    if [ -d "docs" ]; then
        if [ ! -d "99-archive/docs-backup" ]; then
            mv docs 99-archive/docs-backup
            print_success "Moved duplicate docs to archive"
        fi
    fi
}

cleanup_build_artifacts() {
    print_step "Cleaning build artifacts..."

    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    print_success "Removed Python cache files"

    # Clean logs (keep directory, remove old logs)
    if [ -d "logs" ]; then
        find logs -type f -name "*.log" -mtime +7 -delete 2>/dev/null || true
        print_success "Cleaned old log files (>7 days)"
    fi
}

organize_root_docs() {
    print_step "Organizing root-level documentation..."

    # Create a docs-root directory for misc docs
    mkdir -p 00-documentation/00-root-docs

    # Move cleanup/orchestration docs
    for doc in CLEANUP_QUICKSTART.md ORCHESTRATION_*.md ROOT_CLEANUP_PLAN.md; do
        if [ -f "$doc" ]; then
            mv "$doc" 00-documentation/00-root-docs/ 2>/dev/null || true
        fi
    done

    # Move technical reports
    for doc in CORRECTED_PERFORMANCE_CALCULATION.md DSMIL_COMPATIBILITY_REPORT.md LAT5150_ENVIRONMENTS_COMPARISON.md; do
        if [ -f "$doc" ]; then
            mv "$doc" 00-documentation/00-root-docs/ 2>/dev/null || true
        fi
    done

    print_success "Organized root-level documentation"
}

organize_scripts() {
    print_step "Organizing scripts..."

    # Move old orchestration scripts to archive
    for script in orchestrate-completion.sh safe-delete-root-artifacts.sh; do
        if [ -f "$script" ]; then
            mv "$script" 99-archive/ 2>/dev/null || true
        fi
    done

    print_success "Organized scripts"
}

create_directory_structure_doc() {
    print_step "Creating directory structure documentation..."

    cat > STRUCTURE.md << 'EOF'
# DSMIL Unified AI Platform - Directory Structure

## Core Directories

### `00-documentation/`
Complete documentation for the DSMIL platform
- `00-indexes/` - Index files and catalogs
- `01-planning/` - Project planning documents
- `02-analysis/` - Technical analysis
- `03-ai-framework/` - AI framework documentation
- `04-progress/` - Progress reports and changelogs
- `05-reference/` - Reference materials
- `00-root-docs/` - Miscellaneous root-level docs
- `archive/` - Historical documentation

### `01-source/`
Original DSMIL framework source code
- `kernel/` - Kernel module source
- `kernel-driver/` - Kernel driver
- `userspace-tools/` - Userspace utilities
- `systemd/` - Systemd integration
- `tests/` - Test suites
- `security_chaos_framework/` - Security testing framework

### `02-ai-engine/`
AI inference engine and model management
- `dsmil_ai_engine.py` - Main AI engine
- `smart_router.py` - Smart model routing
- `code_specialist.py` - Code generation specialist
- `local_claude_code.py` - Local code editing
- `web_search.py` - Web search integration
- `unified_orchestrator.py` - Multi-backend orchestration
- `sub_agents/` - Specialized sub-agents

### `03-web-interface/`
Web-based user interface
- `clean_ui_v3.html` - Modern ChatGPT-style UI
- `dsmil_unified_server.py` - Backend server
- `military_terminal_v2.html` - Alternative terminal UI
- Documentation for RAG and web features

### `03-security/`
Security documentation and procedures
- `audit/` - Security audit reports
- `procedures/` - Safety and security procedures
- Covert Edition documentation

### `04-integrations/`
External integrations and tools
- `rag_manager.py` - RAG knowledge base
- `web_scraper.py` - Intelligent web crawler
- `crawl4ai_wrapper.py` - Industrial crawler integration

### `05-deployment/`
Deployment configuration and scripts
- `systemd/` - Systemd service files
- `npu-covert-edition.env` - Covert Edition environment
- `verify_system.sh` - System verification

### `99-archive/`
Archived code, old versions, and backups
- Historical versions
- Cleanup backups
- Deprecated code

## Build and Packaging

### `build/`
Build artifacts and compiled binaries

### `packaging/`
Debian packages and distribution files
- `debian/` - Debian packaging
- `dkms/` - DKMS module packaging

### `tpm2_compat/`
TPM 2.0 compatibility layer
- `core/` - Core TPM functionality
- `tools/` - TPM utilities

## Root Files

### Installation
- `install.sh` - Automated installer
- `uninstall.sh` - Uninstaller
- `INSTALL.md` - Installation guide

### Documentation
- `README.md` - Main documentation
- `STRUCTURE.md` - This file

### Configuration
- `.gitignore` - Git ignore patterns
- `DSMIL_UNIVERSAL_FRAMEWORK.py` - Universal framework

### Logs
- `logs/` - Application logs
- `health_log.jsonl` - Health monitoring logs

---

**Last Updated:** $(date +%Y-%m-%d)
**Version:** 8.3
EOF

    print_success "Created STRUCTURE.md"
}

update_gitignore() {
    print_step "Updating .gitignore..."

    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Logs
*.log
logs/*.log
health_log.jsonl

# Build artifacts
build/
dist/
*.egg-info/
.eggs/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Local config
config.local.json
*.env.local

# Models (too large for git)
*.bin
*.gguf
models/

# RAG index
rag_index/
.local/

# Temporary files
tmp/
temp/
*.tmp

# Claude artifacts
.claude/
.claude.json*
.claude_*

# Core dumps
core
core.*
EOF

    print_success "Updated .gitignore"
}

generate_cleanup_report() {
    print_step "Generating cleanup report..."

    cat > CLEANUP_REPORT.md << EOF
# Cleanup Report - $(date +%Y-%m-%d)

## Actions Taken

### Consolidated Directories
- ✓ Merged deployment directories into 05-deployment/
- ✓ Archived duplicate docs/ directory
- ✓ Moved 02-deployment to archive
- ✓ Moved deployment/ to archive

### Cleaned Files
- ✓ Removed Python cache (__pycache__, *.pyc)
- ✓ Cleaned old log files (>7 days)
- ✓ Organized root-level documentation

### Organized Structure
- ✓ Created 00-documentation/00-root-docs/ for misc docs
- ✓ Moved technical reports to documentation
- ✓ Archived old scripts

### Documentation
- ✓ Created STRUCTURE.md (directory layout)
- ✓ Updated .gitignore
- ✓ Generated this cleanup report

## Current Structure

\`\`\`
LAT5150DRVMIL/
├── 00-documentation/     # All documentation
├── 01-source/            # DSMIL framework source
├── 02-ai-engine/         # AI inference engine
├── 03-web-interface/     # Web UI and server
├── 03-security/          # Security docs
├── 04-integrations/      # External integrations
├── 05-deployment/        # Deployment configs
├── 99-archive/           # Archived content
├── build/                # Build artifacts
├── packaging/            # Debian packages
├── tpm2_compat/          # TPM compatibility
├── install.sh            # Installer
├── uninstall.sh          # Uninstaller
├── README.md             # Main docs
└── INSTALL.md            # Install guide
\`\`\`

## Files Archived

The following directories were moved to 99-archive/:
- 02-deployment/ → 99-archive/02-deployment-backup/
- deployment/ → 99-archive/deployment-backup/
- docs/ → 99-archive/docs-backup/
- orchestrate-completion.sh → 99-archive/
- safe-delete-root-artifacts.sh → 99-archive/

## Next Steps

1. Review 99-archive/ and delete if no longer needed
2. Test installation with \`./install.sh\`
3. Verify all features work
4. Commit changes to git

---

**Cleanup completed:** $(date)
**Version:** 8.3
EOF

    print_success "Generated CLEANUP_REPORT.md"
}

main() {
    cd "$SCRIPT_DIR"

    print_banner

    echo -e "${YELLOW}This will organize and clean up the codebase.${NC}"
    echo -e "${YELLOW}Duplicate directories will be archived to 99-archive/${NC}\n"

    read -p "Continue with cleanup? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Cleanup cancelled"
        exit 0
    fi

    # Run cleanup steps
    cleanup_deployment_dirs
    cleanup_docs_dirs
    cleanup_build_artifacts
    organize_root_docs
    organize_scripts
    create_directory_structure_doc
    update_gitignore
    generate_cleanup_report

    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  CLEANUP COMPLETE!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}Summary:${NC}"
    echo -e "  ✓ Consolidated deployment directories"
    echo -e "  ✓ Organized documentation"
    echo -e "  ✓ Cleaned build artifacts"
    echo -e "  ✓ Created STRUCTURE.md"
    echo -e "  ✓ Updated .gitignore\n"

    echo -e "${CYAN}Review:${NC}"
    echo -e "  - Check ${YELLOW}CLEANUP_REPORT.md${NC} for details"
    echo -e "  - Check ${YELLOW}STRUCTURE.md${NC} for directory layout"
    echo -e "  - Review ${YELLOW}99-archive/${NC} and delete if not needed\n"

    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "  1. Test: ${YELLOW}./install.sh${NC}"
    echo -e "  2. Git status: ${YELLOW}git status${NC}"
    echo -e "  3. Commit changes: ${YELLOW}git add -A && git commit -m 'Cleanup and organize codebase'${NC}\n"
}

main "$@"
