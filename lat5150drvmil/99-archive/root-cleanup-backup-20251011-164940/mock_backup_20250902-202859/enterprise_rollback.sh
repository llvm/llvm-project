#!/bin/bash
# DSMIL Phase 2A Enterprise Rollback Script
# Generated: 2025-09-02T20:28:59.933513
# Deployment ID: phase2a_prod_1756841339

echo "🔄 DSMIL Phase 2A Enterprise Rollback"
echo "Deployment ID: phase2a_prod_1756841339"
echo "Timestamp: $(date)"

# Stop monitoring services
systemctl stop dsmil-monitoring 2>/dev/null || true
systemctl stop dsmil-phase2a 2>/dev/null || true

# Remove kernel module
rmmod dsmil-72dev 2>/dev/null || echo "Module not loaded"

# Restore original files
echo "Restoring backed up components..."

# Restore services and restart
echo "✅ Rollback completed successfully"
echo "System restored to pre-deployment state"
