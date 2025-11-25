#!/usr/bin/env python3
"""
Register Xen Hypervisor Management Page
Integrates Xen orchestrator with tactical web UI
"""

import sys
from pathlib import Path
from datetime import datetime

# Add web interface to path
sys.path.insert(0, str(Path(__file__).parent.parent / "03-web-interface"))

from dynamic_page_api import (
    get_page_registry,
    PageRegistration,
    PageMetadata,
    PageEndpoint
)


def register_xen_hypervisor_page():
    """Register Xen hypervisor management page"""

    registry = get_page_registry()

    #=========================================================================
    # Metadata
    # ========================================================================

    metadata = PageMetadata(
        page_id="xen_hypervisor",
        title="Xen Hypervisor Management",
        category="operations",
        icon="⚡",
        route="/hypervisor",
        description="On-demand VM provisioning with NPU/iGPU/NCS2 passthrough and full LAT5150 integration",
        security_classification="CUI",
        tempest_compliant=True,
        registered_by="xen_orchestrator_module",
        registered_at=datetime.utcnow().isoformat(),
        version="1.0.0",
        tactical_mode="comfort"
    )

    # ========================================================================
    # TEMPEST-Compliant HTML Content
    # ========================================================================

    html_content = """
    <div class="tactical-container">

        <!-- System Status Overview -->
        <div class="tactical-grid">
            <div class="tactical-card">
                <div class="tactical-card-header">Orchestrator Status</div>
                <div class="tactical-card-value" id="orchestrator-status">ONLINE</div>
                <span class="tactical-status tactical-status-success" id="orchestrator-badge">OPERATIONAL</span>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">Running VMs</div>
                <div class="tactical-card-value" id="running-vms">0</div>
                <div class="tactical-card-footer">
                    <span id="total-vms">0 registered</span>
                </div>
            </div>

            <div class="tactical-card">
                <div class="tactical-card-header">AI Acceleration</div>
                <div class="tactical-card-value" id="ai-tops">100 TOPS</div>
                <div class="tactical-card-footer">
                    NPU + iGPU + NCS2
                </div>
            </div>
        </div>

        <!-- VM Management Section -->
        <div class="tactical-section">
            <h2 class="tactical-section-title">Virtual Machines</h2>

            <!-- VM Creation -->
            <details class="tactical-collapse" style="margin-bottom: var(--spacing-md);">
                <summary class="tactical-collapse-header" style="cursor: pointer; padding: var(--spacing-sm); background: var(--bg-tertiary); border: 1px solid var(--border-primary); border-radius: 3px; font-weight: 600;">
                    ▶ Create New VM
                </summary>
                <div class="tactical-collapse-content" style="padding: var(--spacing-md); border: 1px solid var(--border-primary); border-top: none; border-radius: 0 0 3px 3px;">
                    <div class="tactical-flex tactical-flex-col tactical-gap-md">
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--spacing-md);">
                            <div>
                                <label class="tactical-label">VM Name</label>
                                <input type="text" id="vm-name" class="tactical-input" placeholder="analysis-vm-01" />
                            </div>

                            <div>
                                <label class="tactical-label">Classification</label>
                                <select id="vm-classification" class="tactical-select">
                                    <option value="UNCLASSIFIED">UNCLASSIFIED</option>
                                    <option value="CUI" selected>CUI</option>
                                    <option value="SECRET">SECRET</option>
                                    <option value="TOP_SECRET">TOP SECRET</option>
                                </select>
                            </div>

                            <div>
                                <label class="tactical-label">vCPUs</label>
                                <input type="number" id="vm-vcpus" class="tactical-input" value="4" min="1" max="64" />
                            </div>

                            <div>
                                <label class="tactical-label">Memory (MB)</label>
                                <input type="number" id="vm-memory" class="tactical-input" value="4096" min="512" step="512" />
                            </div>
                        </div>

                        <div>
                            <label class="tactical-label">Disk Path</label>
                            <input type="text" id="vm-disk" class="tactical-input" placeholder="/opt/lat5150/hypervisor/storage/vm-01.img" />
                        </div>

                        <div style="display: flex; gap: var(--spacing-sm); flex-wrap: wrap;">
                            <label style="display: flex; align-items: center; gap: var(--spacing-xs); color: var(--text-primary);">
                                <input type="checkbox" id="vm-tpm" />
                                <span>TPM 2.0</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: var(--spacing-xs); color: var(--text-primary);">
                                <input type="checkbox" id="vm-secure-boot" />
                                <span>Secure Boot</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: var(--spacing-xs); color: var(--text-primary);">
                                <input type="checkbox" id="vm-encrypted" />
                                <span>Encrypted</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: var(--spacing-xs); color: var(--text-primary);">
                                <input type="checkbox" id="vm-npu" />
                                <span>NPU Passthrough</span>
                            </label>
                            <label style="display: flex; align-items: center; gap: var(--spacing-xs); color: var(--text-primary);">
                                <input type="checkbox" id="vm-gpu" />
                                <span>iGPU Passthrough</span>
                            </label>
                        </div>

                        <div>
                            <button class="tactical-btn tactical-btn-primary" onclick="createVM()">
                                Create VM
                            </button>
                        </div>
                    </div>
                </div>
            </details>

            <!-- VM List -->
            <div id="vm-list-container">
                <table class="tactical-table">
                    <thead>
                        <tr>
                            <th>Name</th>
                            <th>State</th>
                            <th>vCPUs</th>
                            <th>Memory</th>
                            <th>Uptime</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="vm-list-body">
                        <tr>
                            <td colspan="6" style="text-align: center; color: var(--text-muted);">
                                No VMs running. Create a VM to get started.
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- AI Acceleration Passthrough -->
        <div class="tactical-section">
            <h2 class="tactical-section-title">AI Acceleration Passthrough</h2>

            <div class="tactical-grid">
                <div class="tactical-card" id="npu-card">
                    <div class="tactical-card-header">Intel NPU</div>
                    <div class="tactical-card-value" id="npu-tops">30 TOPS</div>
                    <div class="tactical-card-footer">
                        <span class="tactical-status tactical-status-info" id="npu-status">AVAILABLE</span>
                    </div>
                </div>

                <div class="tactical-card" id="gpu-card">
                    <div class="tactical-card-header">Intel iGPU</div>
                    <div class="tactical-card-value" id="gpu-tops">40 TOPS</div>
                    <div class="tactical-card-footer">
                        <span class="tactical-status tactical-status-info" id="gpu-status">AVAILABLE</span>
                    </div>
                </div>

                <div class="tactical-card" id="ncs2-card">
                    <div class="tactical-card-header">Intel NCS2</div>
                    <div class="tactical-card-value" id="ncs2-tops">30 TOPS</div>
                    <div class="tactical-card-footer">
                        <span id="ncs2-count">3 sticks</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- PCI Devices -->
        <details class="tactical-collapse">
            <summary class="tactical-collapse-header" style="cursor: pointer; padding: var(--spacing-sm); background: var(--bg-secondary); border: 1px solid var(--border-primary); border-radius: 3px; font-weight: 600;">
                ▶ Available PCI Devices
            </summary>
            <div class="tactical-collapse-content" style="padding: var(--spacing-md); border: 1px solid var(--border-primary); border-top: none; border-radius: 0 0 3px 3px;">
                <table class="tactical-table" id="pci-devices-table">
                    <thead>
                        <tr>
                            <th>Address</th>
                            <th>Vendor</th>
                            <th>Device</th>
                            <th>Description</th>
                        </tr>
                    </thead>
                    <tbody id="pci-devices-body">
                        <tr>
                            <td colspan="4" style="text-align: center; color: var(--text-muted);">
                                Loading PCI devices...
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </details>

    </div>

    <script>
    /**
     * Xen Hypervisor Management Client
     * TEMPEST-compliant: No animations, instant updates
     */

    // Initialize on load
    (function() {
        refreshStatus();
        refreshVMList();
        refreshAIAcceleration();
        refreshPCIDevices();

        // Auto-refresh every 5 seconds
        setInterval(refreshStatus, 5000);
        setInterval(refreshVMList, 5000);
    })();

    // Refresh orchestrator status
    async function refreshStatus() {
        try {
            const response = await fetch('/api/hypervisor/status');
            const data = await response.json();

            document.getElementById('running-vms').textContent = data.running_vms || 0;
            document.getElementById('total-vms').textContent = `${data.registered_vms || 0} registered`;
            document.getElementById('ai-tops').textContent = `${data.total_ai_tops || 0} TOPS`;

        } catch (error) {
            console.error('Failed to refresh status:', error);
        }
    }

    // Refresh VM list
    async function refreshVMList() {
        try {
            const response = await fetch('/api/hypervisor/vms');
            const data = await response.json();

            const tbody = document.getElementById('vm-list-body');

            if (!data.vms || data.vms.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="6" style="text-align: center; color: var(--text-muted);">
                            No VMs running. Create a VM to get started.
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = data.vms.map(vm => `
                <tr>
                    <td><strong>${escapeHtml(vm.name)}</strong></td>
                    <td><span class="tactical-status tactical-status-${getStateClass(vm.state)}">${vm.state}</span></td>
                    <td>${vm.vcpus}</td>
                    <td>${vm.memory_mb} MB</td>
                    <td>${formatUptime(vm.uptime_seconds)}</td>
                    <td>
                        ${vm.state === 'running' ?
                            `<button class="tactical-btn tactical-btn-danger" style="padding: 4px 8px; font-size: 11px;" onclick="stopVM('${vm.name}')">Stop</button>` :
                            `<button class="tactical-btn tactical-btn-primary" style="padding: 4px 8px; font-size: 11px;" onclick="startVM('${vm.name}')">Start</button>`
                        }
                    </td>
                </tr>
            `).join('');

        } catch (error) {
            console.error('Failed to refresh VM list:', error);
        }
    }

    // Refresh AI acceleration info
    async function refreshAIAcceleration() {
        try {
            const response = await fetch('/api/hypervisor/ai-acceleration');
            const data = await response.json();

            if (data.error) {
                return;
            }

            data.components.forEach(component => {
                if (component.type === 'NPU') {
                    document.getElementById('npu-tops').textContent = `${component.tops} TOPS`;
                    document.getElementById('npu-status').textContent = component.passthrough_available ? 'AVAILABLE' : 'IN USE';
                } else if (component.type === 'iGPU') {
                    document.getElementById('gpu-tops').textContent = `${component.tops} TOPS`;
                    document.getElementById('gpu-status').textContent = component.passthrough_available ? 'AVAILABLE' : 'IN USE';
                } else if (component.type === 'NCS2') {
                    document.getElementById('ncs2-tops').textContent = `${component.total_tops} TOPS`;
                    document.getElementById('ncs2-count').textContent = `${component.count} sticks`;
                }
            });

        } catch (error) {
            console.error('Failed to refresh AI acceleration:', error);
        }
    }

    // Refresh PCI devices
    async function refreshPCIDevices() {
        try {
            const response = await fetch('/api/hypervisor/pci-devices');
            const data = await response.json();

            const tbody = document.getElementById('pci-devices-body');

            if (!data.devices || data.devices.length === 0) {
                tbody.innerHTML = `
                    <tr>
                        <td colspan="4" style="text-align: center; color: var(--text-muted);">
                            No PCI devices available for passthrough
                        </td>
                    </tr>
                `;
                return;
            }

            tbody.innerHTML = data.devices.map(dev => `
                <tr>
                    <td><code>${dev.domain}:${dev.bus}:${dev.slot}.${dev.function}</code></td>
                    <td>${dev.vendor_id}</td>
                    <td>${dev.device_id}</td>
                    <td>${escapeHtml(dev.description)}</td>
                </tr>
            `).join('');

        } catch (error) {
            console.error('Failed to refresh PCI devices:', error);
        }
    }

    // Create VM
    async function createVM() {
        const vmData = {
            name: document.getElementById('vm-name').value,
            vcpus: parseInt(document.getElementById('vm-vcpus').value),
            memory_mb: parseInt(document.getElementById('vm-memory').value),
            classification: document.getElementById('vm-classification').value,
            disk_path: document.getElementById('vm-disk').value,
            tpm_enabled: document.getElementById('vm-tpm').checked,
            secure_boot: document.getElementById('vm-secure-boot').checked,
            encrypted: document.getElementById('vm-encrypted').checked,
            npu_passthrough: document.getElementById('vm-npu').checked,
            gpu_passthrough: document.getElementById('vm-gpu').checked
        };

        if (!vmData.name || !vmData.disk_path) {
            alert('Please fill in all required fields');
            return;
        }

        try {
            const response = await fetch('/api/hypervisor/create-vm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(vmData)
            });

            const result = await response.json();

            if (result.success) {
                alert(`VM '${vmData.name}' created successfully!`);
                // Clear form
                document.getElementById('vm-name').value = '';
                document.getElementById('vm-disk').value = '';
                // Refresh lists
                refreshStatus();
                refreshVMList();
            } else {
                alert(`Failed to create VM: ${result.error}`);
            }

        } catch (error) {
            console.error('Failed to create VM:', error);
            alert('Failed to create VM: ' + error.message);
        }
    }

    // Start VM
    async function startVM(name) {
        try {
            const response = await fetch('/api/hypervisor/start-vm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name})
            });

            const result = await response.json();

            if (result.success) {
                // Refresh immediately
                setTimeout(() => {
                    refreshStatus();
                    refreshVMList();
                }, 1000);
            } else {
                alert(`Failed to start VM: ${result.error}`);
            }

        } catch (error) {
            console.error('Failed to start VM:', error);
            alert('Failed to start VM: ' + error.message);
        }
    }

    // Stop VM
    async function stopVM(name) {
        if (!confirm(`Stop VM '${name}'?`)) {
            return;
        }

        try {
            const response = await fetch('/api/hypervisor/stop-vm', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({name, force: false})
            });

            const result = await response.json();

            if (result.success) {
                // Refresh immediately
                setTimeout(() => {
                    refreshStatus();
                    refreshVMList();
                }, 1000);
            } else {
                alert(`Failed to stop VM: ${result.error}`);
            }

        } catch (error) {
            console.error('Failed to stop VM:', error);
            alert('Failed to stop VM: ' + error.message);
        }
    }

    // Utility functions
    function getStateClass(state) {
        const classes = {
            'running': 'success',
            'stopped': 'info',
            'paused': 'warning',
            'error': 'error'
        };
        return classes[state] || 'info';
    }

    function formatUptime(seconds) {
        if (seconds === null || seconds === undefined) {
            return '--';
        }

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else {
            return `${minutes}m`;
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    </script>
    """

    # ========================================================================
    # API Endpoints
    # ========================================================================

    endpoints = [
        # Status endpoint
        PageEndpoint(
            method="GET",
            path="/api/hypervisor/status",
            handler="get_status",
            requires_auth=True,
            rate_limit=100
        ),

        # List VMs
        PageEndpoint(
            method="GET",
            path="/api/hypervisor/vms",
            handler="list_vms",
            requires_auth=True,
            rate_limit=100
        ),

        # Create VM
        PageEndpoint(
            method="POST",
            path="/api/hypervisor/create-vm",
            handler="create_vm",
            requires_auth=True,
            rate_limit=10,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "pattern": "^[a-z0-9_-]+$"},
                    "vcpus": {"type": "integer", "minimum": 1, "maximum": 64},
                    "memory_mb": {"type": "integer", "minimum": 512},
                    "classification": {"enum": ["UNCLASSIFIED", "CUI", "SECRET", "TOP_SECRET"]},
                    "disk_path": {"type": "string"}
                },
                "required": ["name", "vcpus", "memory_mb", "classification", "disk_path"]
            }
        ),

        # Start VM
        PageEndpoint(
            method="POST",
            path="/api/hypervisor/start-vm",
            handler="start_vm",
            requires_auth=True,
            rate_limit=30,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        ),

        # Stop VM
        PageEndpoint(
            method="POST",
            path="/api/hypervisor/stop-vm",
            handler="stop_vm",
            requires_auth=True,
            rate_limit=30,
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "force": {"type": "boolean"}
                },
                "required": ["name"]
            }
        ),

        # AI Acceleration info
        PageEndpoint(
            method="GET",
            path="/api/hypervisor/ai-acceleration",
            handler="get_ai_acceleration",
            requires_auth=True,
            rate_limit=60
        ),

        # PCI Devices
        PageEndpoint(
            method="GET",
            path="/api/hypervisor/pci-devices",
            handler="get_pci_devices",
            requires_auth=True,
            rate_limit=60
        )
    ]

    # ========================================================================
    # Register Page
    # ========================================================================

    registration = PageRegistration(
        metadata=metadata,
        html_content=html_content,
        endpoints=endpoints
    )

    try:
        success = registry.register_page(registration, overwrite=True)
        if success:
            print(f"✓ Successfully registered: {metadata.title}")
            print(f"  Page ID: {metadata.page_id}")
            print(f"  Route: {metadata.route}")
            print(f"  Endpoints: {len(endpoints)}")
            print(f"  TEMPEST Compliant: {metadata.tempest_compliant}")
            print()
            print("Access at: http://localhost:5001/page/xen_hypervisor")
            return True
        else:
            print("✗ Failed to register page")
            return False

    except Exception as e:
        print(f"✗ Registration error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Registering Xen Hypervisor Management Page")
    print("=" * 70)
    print()

    register_xen_hypervisor_page()
