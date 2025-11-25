#!/usr/bin/env python3
"""
Xen Hypervisor API Handlers
Flask endpoints for the Xen management page
"""

import uuid
from flask import jsonify, request
from xen_orchestrator import get_xen_orchestrator, VMConfig, VMClassification, PCIDevice


def register_xen_api_handlers(app):
    """
    Register Xen API handlers with Flask app

    Usage:
        from xen_api_handlers import register_xen_api_handlers
        register_xen_api_handlers(app)
    """

    orchestrator = get_xen_orchestrator()

    @app.route('/api/hypervisor/status', methods=['GET'])
    def get_hypervisor_status():
        """Get orchestrator status"""
        try:
            status = orchestrator.export_status()
            return jsonify(status)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/hypervisor/vms', methods=['GET'])
    def list_vms():
        """List all VMs"""
        try:
            vms = orchestrator.list_vms()
            return jsonify({
                "vms": [
                    {
                        "name": vm.name,
                        "state": vm.state.value,
                        "uuid": vm.uuid,
                        "vcpus": vm.vcpus,
                        "memory_mb": vm.memory_mb,
                        "uptime_seconds": vm.uptime_seconds,
                        "last_updated": vm.last_updated
                    }
                    for vm in vms
                ]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/hypervisor/create-vm', methods=['POST'])
    def create_vm():
        """Create a new VM"""
        try:
            data = request.json

            # Create VM configuration
            config = VMConfig(
                name=data['name'],
                uuid=str(uuid.uuid4()),
                vcpus=data['vcpus'],
                memory_mb=data['memory_mb'],
                classification=VMClassification(data['classification']),
                disk_path=data['disk_path'],
                tpm_enabled=data.get('tpm_enabled', False),
                secure_boot=data.get('secure_boot', False),
                encrypted=data.get('encrypted', False),
                npu_passthrough=data.get('npu_passthrough', False),
                gpu_passthrough=data.get('gpu_passthrough', False)
            )

            # Create VM
            success = orchestrator.create_vm(config)

            if success:
                return jsonify({
                    "success": True,
                    "name": config.name,
                    "uuid": config.uuid
                })
            else:
                return jsonify({
                    "success": False,
                    "error": "Failed to create VM"
                }), 500

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400

    @app.route('/api/hypervisor/start-vm', methods=['POST'])
    def start_vm():
        """Start a VM"""
        try:
            data = request.json
            name = data['name']

            success = orchestrator.start_vm(name)

            return jsonify({
                "success": success,
                "name": name
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400

    @app.route('/api/hypervisor/stop-vm', methods=['POST'])
    def stop_vm():
        """Stop a VM"""
        try:
            data = request.json
            name = data['name']
            force = data.get('force', False)

            success = orchestrator.stop_vm(name, force=force)

            return jsonify({
                "success": success,
                "name": name
            })

        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 400

    @app.route('/api/hypervisor/ai-acceleration', methods=['GET'])
    def get_ai_acceleration():
        """Get AI acceleration summary"""
        try:
            summary = orchestrator.get_ai_acceleration_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/hypervisor/pci-devices', methods=['GET'])
    def get_pci_devices():
        """Get available PCI devices"""
        try:
            devices = orchestrator.get_available_pci_devices()
            return jsonify({
                "devices": [
                    {
                        "domain": dev.domain,
                        "bus": dev.bus,
                        "slot": dev.slot,
                        "function": dev.function,
                        "device_id": dev.device_id,
                        "vendor_id": dev.vendor_id,
                        "description": dev.description,
                        "iommu_group": dev.iommu_group
                    }
                    for dev in devices
                ]
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    print("✓ Registered Xen API handlers")
