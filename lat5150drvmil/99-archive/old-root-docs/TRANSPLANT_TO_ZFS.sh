#!/bin/bash
#═══════════════════════════════════════════════════════════════════════════════
# DSMIL AI Framework - ZFS Transplant Script
# Version: 8.3.2
# Transplants complete AI framework to ZFS encrypted environment
#═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# Configuration
SUDO_PASS="1786"
ZFS_PASS="1/0523/600260"
ZFS_POOL="rpool"
ZFS_BE="livecd-xen-ai"  # From your transplant session
CURRENT_AI_DIR="/home/john/LAT5150DRVMIL"

print_banner() {
    echo -e "${CYAN}"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  DSMIL AI FRAMEWORK → ZFS TRANSPLANT"
    echo "  Version: 8.3.2"
    echo "  Target: $ZFS_POOL/ROOT/$ZFS_BE (Encrypted)"
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo -e "${NC}"
}

print_section() {
    echo -e "\n${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_step() {
    echo -e "\n${BLUE}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

#═══════════════════════════════════════════════════════════════════════════════
# Pre-Flight Checks
#═══════════════════════════════════════════════════════════════════════════════

check_prerequisites() {
    print_section "PRE-FLIGHT CHECKS"

    # Check if running on current system (not in transplanted BE yet)
    if [ "$(uname -r)" = "6.16.12-xen-ai-hardened" ]; then
        print_error "Already booted into transplanted system!"
        print_warning "This script should be run BEFORE transplanting to prepare the AI framework."
        exit 1
    fi

    # Check ZFS availability
    if ! command -v zfs >/dev/null 2>&1; then
        print_error "ZFS not available on this system"
        print_warning "You may be in LiveCD mode. Import your pool first:"
        echo "  sudo zpool import $ZFS_POOL"
        echo "  echo '$ZFS_PASS' | sudo zfs load-key $ZFS_POOL"
        exit 1
    fi

    # Check if AI framework exists
    if [ ! -d "$CURRENT_AI_DIR" ]; then
        print_error "AI framework not found at: $CURRENT_AI_DIR"
        exit 1
    fi

    # Check framework size
    AI_SIZE=$(du -sh "$CURRENT_AI_DIR" 2>/dev/null | awk '{print $1}')
    print_success "AI framework found: $AI_SIZE"

    # Check for Ollama models
    if [ -d "$HOME/.ollama" ]; then
        OLLAMA_SIZE=$(du -sh "$HOME/.ollama" 2>/dev/null | awk '{print $1}')
        print_success "Ollama models found: $OLLAMA_SIZE"
    else
        print_warning "Ollama models not found (will skip)"
    fi

    # Check for RAG index
    if [ -d "$HOME/.local/share/dsmil" ]; then
        RAG_SIZE=$(du -sh "$HOME/.local/share/dsmil" 2>/dev/null | awk '{print $1}')
        print_success "RAG index found: $RAG_SIZE"
    else
        print_warning "RAG index not found (will skip)"
    fi

    print_success "Pre-flight checks complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# ZFS Pool Management
#═══════════════════════════════════════════════════════════════════════════════

setup_zfs_pool() {
    print_section "ZFS POOL SETUP"

    # Check if pool is imported
    if ! zpool list "$ZFS_POOL" >/dev/null 2>&1; then
        print_step "Importing ZFS pool: $ZFS_POOL..."
        echo "$SUDO_PASS" | sudo -S zpool import "$ZFS_POOL" || {
            print_error "Failed to import pool"
            exit 1
        }
    fi

    # Load encryption key
    if ! zfs get keystatus "$ZFS_POOL" | grep -q "available"; then
        print_step "Loading encryption key..."
        echo "$ZFS_PASS" | echo "$SUDO_PASS" | sudo -S zfs load-key "$ZFS_POOL"
    fi

    print_success "ZFS pool ready: $ZFS_POOL"
}

create_ai_datasets() {
    print_section "CREATING AI FRAMEWORK DATASETS"

    print_step "Creating ZFS datasets for AI framework..."

    # Create main AI dataset
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework" 2>/dev/null || {
        print_warning "Dataset exists, using existing"
    }

    # Set properties for AI framework
    echo "$SUDO_PASS" | sudo -S zfs set compression=lz4 "$ZFS_POOL/ai-framework"
    echo "$SUDO_PASS" | sudo -S zfs set atime=off "$ZFS_POOL/ai-framework"

    # Create sub-datasets with optimized settings
    print_step "Creating optimized sub-datasets..."

    # Source code dataset
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework/source" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs set recordsize=128k "$ZFS_POOL/ai-framework/source"

    # Ollama models dataset (large files)
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework/ollama-models" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs set recordsize=1M "$ZFS_POOL/ai-framework/ollama-models"
    echo "$SUDO_PASS" | sudo -S zfs set compression=lz4 "$ZFS_POOL/ai-framework/ollama-models"

    # RAG index dataset (small files, high compression)
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework/rag-index" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs set recordsize=16k "$ZFS_POOL/ai-framework/rag-index"
    echo "$SUDO_PASS" | sudo -S zfs set compression=zstd "$ZFS_POOL/ai-framework/rag-index"

    # Configuration dataset
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework/config" 2>/dev/null || true

    # Logs dataset (high compression)
    echo "$SUDO_PASS" | sudo -S zfs create -p "$ZFS_POOL/ai-framework/logs" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs set recordsize=128k "$ZFS_POOL/ai-framework/logs"
    echo "$SUDO_PASS" | sudo -S zfs set compression=gzip-9 "$ZFS_POOL/ai-framework/logs"

    # Set mountpoints
    echo "$SUDO_PASS" | sudo -S zfs set mountpoint=/opt/dsmil "$ZFS_POOL/ai-framework/source"
    echo "$SUDO_PASS" | sudo -S zfs set mountpoint=/var/lib/ollama/models "$ZFS_POOL/ai-framework/ollama-models"
    echo "$SUDO_PASS" | sudo -S zfs set mountpoint=/var/lib/dsmil/rag "$ZFS_POOL/ai-framework/rag-index"
    echo "$SUDO_PASS" | sudo -S zfs set mountpoint=/etc/dsmil "$ZFS_POOL/ai-framework/config"
    echo "$SUDO_PASS" | sudo -S zfs set mountpoint=/var/log/dsmil "$ZFS_POOL/ai-framework/logs"

    print_success "ZFS datasets created and optimized"
}

#═══════════════════════════════════════════════════════════════════════════════
# Data Transplant
#═══════════════════════════════════════════════════════════════════════════════

create_safety_snapshot() {
    print_section "CREATING SAFETY SNAPSHOT"

    local snapshot_name="$ZFS_POOL@before-ai-transplant-$(date +%Y%m%d-%H%M)"

    print_step "Creating recursive snapshot: $snapshot_name"
    echo "$SUDO_PASS" | sudo -S zfs snapshot -r "$snapshot_name"

    print_success "Safety snapshot created: $snapshot_name"
    print_warning "Rollback command if needed: sudo zfs rollback -r $snapshot_name"
}

transplant_ai_framework() {
    print_section "TRANSPLANTING AI FRAMEWORK"

    # Mount datasets
    print_step "Mounting ZFS datasets..."
    echo "$SUDO_PASS" | sudo -S zfs mount "$ZFS_POOL/ai-framework/source" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs mount "$ZFS_POOL/ai-framework/ollama-models" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs mount "$ZFS_POOL/ai-framework/rag-index" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs mount "$ZFS_POOL/ai-framework/config" 2>/dev/null || true
    echo "$SUDO_PASS" | sudo -S zfs mount "$ZFS_POOL/ai-framework/logs" 2>/dev/null || true

    # Create directories
    echo "$SUDO_PASS" | sudo -S mkdir -p /opt/dsmil /var/lib/ollama/models /var/lib/dsmil/rag /etc/dsmil /var/log/dsmil

    # Copy AI framework source
    print_step "Copying AI framework source..."
    echo "$SUDO_PASS" | sudo -S rsync -avh --info=progress2 "$CURRENT_AI_DIR/" /opt/dsmil/
    print_success "AI framework copied: $(du -sh /opt/dsmil | awk '{print $1}')"

    # Copy Ollama models if present
    if [ -d "$HOME/.ollama" ]; then
        print_step "Copying Ollama models..."
        echo "$SUDO_PASS" | sudo -S rsync -avh --info=progress2 "$HOME/.ollama/" /var/lib/ollama/
        print_success "Ollama models copied"
    fi

    # Copy RAG index if present
    if [ -d "$HOME/.local/share/dsmil" ]; then
        print_step "Copying RAG index..."
        echo "$SUDO_PASS" | sudo -S rsync -avh --info=progress2 "$HOME/.local/share/dsmil/" /var/lib/dsmil/rag/
        print_success "RAG index copied"
    fi

    # Copy configuration
    if [ -d "$HOME/.config/dsmil" ]; then
        print_step "Copying configuration..."
        echo "$SUDO_PASS" | sudo -S cp -r "$HOME/.config/dsmil/"* /etc/dsmil/ 2>/dev/null || true
        print_success "Configuration copied"
    fi

    # Set permissions
    echo "$SUDO_PASS" | sudo -S chown -R $USER:$USER /opt/dsmil
    echo "$SUDO_PASS" | sudo -S chown -R ollama:ollama /var/lib/ollama 2>/dev/null || echo "$SUDO_PASS" | sudo -S chown -R $USER:$USER /var/lib/ollama
    echo "$SUDO_PASS" | sudo -S chown -R $USER:$USER /var/lib/dsmil
    echo "$SUDO_PASS" | sudo -S chown -R $USER:$USER /etc/dsmil
    echo "$SUDO_PASS" | sudo -S chown -R $USER:$USER /var/log/dsmil

    print_success "Data transplant complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# System Integration
#═══════════════════════════════════════════════════════════════════════════════

update_systemd_service() {
    print_section "UPDATING SYSTEMD SERVICE"

    print_step "Creating systemd service for ZFS installation..."

    echo "$SUDO_PASS" | sudo -S tee /etc/systemd/system/dsmil-server.service > /dev/null << EOF
[Unit]
Description=DSMIL Unified AI Server
After=network.target ollama.service zfs-mount.service
Wants=ollama.service
Requires=zfs-mount.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/dsmil/03-web-interface
ExecStart=/usr/bin/python3 /opt/dsmil/03-web-interface/dsmil_unified_server.py
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security
PrivateTmp=true
NoNewPrivileges=true

# Environment
Environment="PATH=/usr/local/bin:/usr/bin:/bin"
Environment="DSMIL_HOME=/opt/dsmil"

[Install]
WantedBy=multi-user.target
EOF

    echo "$SUDO_PASS" | sudo -S systemctl daemon-reload
    echo "$SUDO_PASS" | sudo -S systemctl enable dsmil-server.service

    print_success "Systemd service configured"
}

update_configuration() {
    print_section "UPDATING CONFIGURATION"

    # Update config.json paths
    if [ -f "/etc/dsmil/config.json" ]; then
        print_step "Updating configuration paths..."

        echo "$SUDO_PASS" | sudo -S sed -i 's|/home/john/LAT5150DRVMIL|/opt/dsmil|g' /etc/dsmil/config.json
        echo "$SUDO_PASS" | sudo -S sed -i 's|/home/john/.local/share/dsmil/rag_index|/var/lib/dsmil/rag|g' /etc/dsmil/config.json

        print_success "Configuration updated"
    fi

    # Create symlinks for compatibility
    print_step "Creating compatibility symlinks..."
    ln -sf /opt/dsmil "$HOME/LAT5150DRVMIL-zfs" 2>/dev/null || true
    ln -sf /var/lib/ollama "$HOME/.ollama-zfs" 2>/dev/null || true
    ln -sf /var/lib/dsmil/rag "$HOME/.local/share/dsmil-zfs" 2>/dev/null || true

    print_success "Symlinks created"
}

#═══════════════════════════════════════════════════════════════════════════════
# ZFS Optimization
#═══════════════════════════════════════════════════════════════════════════════

optimize_zfs_datasets() {
    print_section "OPTIMIZING ZFS DATASETS"

    print_step "Creating initial snapshots..."
    echo "$SUDO_PASS" | sudo -S zfs snapshot "$ZFS_POOL/ai-framework/source@initial-transplant"
    echo "$SUDO_PASS" | sudo -S zfs snapshot "$ZFS_POOL/ai-framework/ollama-models@initial-transplant"
    echo "$SUDO_PASS" | sudo -S zfs snapshot "$ZFS_POOL/ai-framework/rag-index@initial-transplant"

    print_step "Checking compression ratios..."
    for dataset in source ollama-models rag-index logs; do
        ratio=$(zfs get compressratio "$ZFS_POOL/ai-framework/$dataset" -H -o value 2>/dev/null || echo "N/A")
        echo "  $dataset: $ratio"
    done

    print_success "ZFS optimization complete"
}

#═══════════════════════════════════════════════════════════════════════════════
# Integration with livecd-gen
#═══════════════════════════════════════════════════════════════════════════════

integrate_with_livecd() {
    print_section "INTEGRATING WITH LIVECD-GEN SYSTEM"

    # Create integration script in livecd-gen
    if [ -d "/home/john/livecd-gen" ]; then
        print_step "Creating livecd-gen integration module..."

        cat > /tmp/dsmil_ai_integration.sh << 'INTEGRATION_EOF'
#!/bin/bash
# DSMIL AI Framework Integration Module for LiveCD

install_dsmil_ai_framework() {
    echo "[+] Installing DSMIL AI Framework..."

    # Copy from ZFS datasets to chroot
    rsync -a /opt/dsmil/ "${CHROOT_DIR}/opt/dsmil/" 2>/dev/null || {
        echo "[!] AI framework not available, skipping"
        return 0
    }

    # Install Python dependencies in chroot
    systemd-nspawn -D "${CHROOT_DIR}" --bind-ro=/etc/resolv.conf \
        /usr/bin/bash -c "pip3 install --break-system-packages \
            requests anthropic flask sentence-transformers faiss-cpu"

    # Copy systemd service
    cp /etc/systemd/system/dsmil-server.service \
       "${CHROOT_DIR}/etc/systemd/system/" 2>/dev/null || true

    # Enable service in chroot
    systemd-nspawn -D "${CHROOT_DIR}" \
        systemctl enable dsmil-server.service 2>/dev/null || true

    echo "[+] DSMIL AI Framework installed to LiveCD"
}
INTEGRATION_EOF

        echo "$SUDO_PASS" | sudo -S mv /tmp/dsmil_ai_integration.sh /home/john/livecd-gen/src/modules/
        echo "$SUDO_PASS" | sudo -S chmod +x /home/john/livecd-gen/src/modules/dsmil_ai_integration.sh

        print_success "Integration module created"
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Verification
#═══════════════════════════════════════════════════════════════════════════════

verify_transplant() {
    print_section "VERIFYING TRANSPLANT"

    local errors=0

    # Check datasets exist
    print_step "Checking ZFS datasets..."
    for dataset in source ollama-models rag-index config logs; do
        if zfs list "$ZFS_POOL/ai-framework/$dataset" >/dev/null 2>&1; then
            print_success "$dataset dataset exists"
        else
            print_error "$dataset dataset missing"
            ((errors++))
        fi
    done

    # Check directories
    print_step "Checking directories..."
    for dir in /opt/dsmil /var/lib/ollama /var/lib/dsmil/rag /etc/dsmil; do
        if [ -d "$dir" ]; then
            size=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
            print_success "$dir exists ($size)"
        else
            print_error "$dir missing"
            ((errors++))
        fi
    done

    # Check service file
    if [ -f "/etc/systemd/system/dsmil-server.service" ]; then
        print_success "Systemd service configured"
    else
        print_error "Systemd service missing"
        ((errors++))
    fi

    if [ $errors -eq 0 ]; then
        print_success "Verification complete - No errors!"
        return 0
    else
        print_error "Verification found $errors errors"
        return 1
    fi
}

#═══════════════════════════════════════════════════════════════════════════════
# Final Instructions
#═══════════════════════════════════════════════════════════════════════════════

print_final_instructions() {
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  AI FRAMEWORK TRANSPLANT COMPLETE!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════════════════════${NC}\n"

    echo -e "${CYAN}📍 AI Framework Location (ZFS):${NC}"
    echo -e "   Source:  ${YELLOW}/opt/dsmil${NC} (on $ZFS_POOL/ai-framework/source)"
    echo -e "   Models:  ${YELLOW}/var/lib/ollama/models${NC} (on $ZFS_POOL/ai-framework/ollama-models)"
    echo -e "   RAG:     ${YELLOW}/var/lib/dsmil/rag${NC} (on $ZFS_POOL/ai-framework/rag-index)"
    echo -e "   Config:  ${YELLOW}/etc/dsmil${NC} (on $ZFS_POOL/ai-framework/config)"
    echo -e "   Logs:    ${YELLOW}/var/lib/dsmil/logs${NC} (on $ZFS_POOL/ai-framework/logs)\n"

    echo -e "${CYAN}📊 ZFS Dataset Info:${NC}"
    zfs list -r "$ZFS_POOL/ai-framework" -o name,used,avail,refer,compressratio 2>/dev/null | head -10

    echo -e "\n${CYAN}📸 Snapshots Created:${NC}"
    echo -e "   Safety: ${YELLOW}$ZFS_POOL@before-ai-transplant-*${NC}"
    echo -e "   Initial: ${YELLOW}$ZFS_POOL/ai-framework/*@initial-transplant${NC}\n"

    echo -e "${CYAN}🔧 Service Management:${NC}"
    echo -e "   Start:   ${YELLOW}sudo systemctl start dsmil-server${NC}"
    echo -e "   Status:  ${YELLOW}sudo systemctl status dsmil-server${NC}"
    echo -e "   Logs:    ${YELLOW}sudo journalctl -u dsmil-server -f${NC}\n"

    echo -e "${CYAN}🌐 Access Interface:${NC}"
    echo -e "   URL: ${YELLOW}http://localhost:9876${NC}\n"

    echo -e "${CYAN}📚 Next Steps:${NC}"
    echo "   1. Start DSMIL service: sudo systemctl start dsmil-server"
    echo "   2. Verify interface: curl http://localhost:9876/status"
    echo "   3. Test in browser: xdg-open http://localhost:9876"
    echo "   4. Integration: Add to livecd-gen if needed"
    echo "   5. Snapshot: sudo zfs snapshot $ZFS_POOL/ai-framework@working"
    echo ""

    echo -e "${CYAN}💡 ZFS Commands:${NC}"
    echo -e "   List datasets: ${YELLOW}zfs list -r $ZFS_POOL/ai-framework${NC}"
    echo -e "   List snapshots: ${YELLOW}zfs list -t snapshot | grep ai-framework${NC}"
    echo -e "   Compression ratio: ${YELLOW}zfs get compressratio $ZFS_POOL/ai-framework${NC}"
    echo -e "   Rollback: ${YELLOW}sudo zfs rollback $ZFS_POOL/ai-framework/source@initial-transplant${NC}\n"

    echo -e "${GREEN}Transplant successful! AI framework now on encrypted ZFS! 🚀${NC}\n"
}

#═══════════════════════════════════════════════════════════════════════════════
# Main Execution
#═══════════════════════════════════════════════════════════════════════════════

main() {
    print_banner

    echo -e "${YELLOW}This will transplant the complete DSMIL AI framework to ZFS:${NC}"
    echo -e "  • Source: $CURRENT_AI_DIR → /opt/dsmil (ZFS)"
    echo -e "  • Models: ~/.ollama → /var/lib/ollama/models (ZFS)"
    echo -e "  • RAG: ~/.local/share/dsmil → /var/lib/dsmil/rag (ZFS)"
    echo -e "  • Config: ~/.config/dsmil → /etc/dsmil (ZFS)"
    echo -e "  • Target pool: $ZFS_POOL (encrypted)\n"

    echo -e "${CYAN}Benefits of ZFS installation:${NC}"
    echo -e "  ✓ Snapshots for instant rollback"
    echo -e "  ✓ Compression saves 30-50% space"
    echo -e "  ✓ Data integrity with checksums"
    echo -e "  ✓ Encrypted storage"
    echo -e "  ✓ Integration with livecd-gen system"
    echo ""

    read -p "Continue with transplant? [Y/n]: " -r
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        print_warning "Transplant cancelled"
        exit 0
    fi

    # Execute transplant
    check_prerequisites
    setup_zfs_pool
    create_safety_snapshot
    create_ai_datasets
    transplant_ai_framework
    update_systemd_service
    update_configuration
    optimize_zfs_datasets
    integrate_with_livecd

    if verify_transplant; then
        print_final_instructions
    else
        print_error "Transplant verification failed!"
        print_warning "Check errors above. Rollback available via ZFS snapshot."
        exit 1
    fi
}

main "$@"
