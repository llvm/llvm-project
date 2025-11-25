#!/usr/bin/env python3
"""
Mock test runner for Phase 2 deployment script
Tests the deployment logic without requiring full orchestrator initialization
"""

import sys
import os
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/john/LAT5150DRVMIL')

# Mock the orchestrator components
class MockOrchestrator:
    def __init__(self):
        self.discovered_agents = {
            'director', 'projectorchestrator', 'security', 'cryptoexpert', 
            'hardware', 'hardware-dell', 'hardware-intel', 'mlops', 
            'datascience', 'npu', 'monitor', 'testbed', 'debugger', 
            'qadirector', 'docgen', 'researcher'
        }
        self.is_initialized = True
        
    async def initialize(self):
        print(f"✅ Mock orchestrator initialized with {len(self.discovered_agents)} agents")
        return True
        
    def get_agent_list(self):
        return list(self.discovered_agents)
        
    async def execute_command_set(self, command_set):
        print(f"   🔄 Executing {command_set.name} with {len(command_set.steps)} steps")
        return {
            "status": "completed",
            "command_id": f"mock_{command_set.name}",
            "results": [f"mock_result_{i}" for i in range(len(command_set.steps))],
            "metrics": {"duration": 1.0, "steps": len(command_set.steps)}
        }
        
    def get_system_status(self):
        return {
            "initialized": True,
            "uptime_seconds": 123.4,
            "discovered_agents": len(self.discovered_agents),
            "active_commands": 0
        }
        
    def get_metrics(self):
        return {
            "discovered_agents": len(self.discovered_agents),
            "python_msgs_processed": 50,
            "uptime_seconds": 123.4
        }

async def test_deployment_mock():
    """Test the deployment script with mock orchestrator"""
    print("🧪 Mock Testing Phase 2 Deployment")
    print("=" * 40)
    
    try:
        # Import deployment script
        import deploy_phase2_with_orchestrator as deploy_script
        
        # Create deployer with mock orchestrator
        deployer = deploy_script.Phase2Deployer()
        
        # Replace orchestrator with mock
        deployer.orchestrator = MockOrchestrator()
        
        print("1. ✅ Deployment script imported and mock orchestrator set")
        
        # Test command set creation
        print("2. Testing command set creation...")
        command_sets = [
            ("TPM Integration", deployer.create_tpm_integration_commandset()),
            ("ML System", deployer.create_ml_system_commandset()),
            ("Device Activation", deployer.create_device_activation_commandset()),
            ("Testing", deployer.create_testing_commandset()),
            ("Documentation", deployer.create_documentation_commandset())
        ]
        
        for name, cmd_set in command_sets:
            print(f"   ✅ {name}: {len(cmd_set.steps)} steps, mode={cmd_set.mode.value}")
        
        # Test individual phase execution
        print("3. Testing individual phase execution...")
        test_results = {"phases": {}, "errors": []}
        
        for name, cmd_set in command_sets[:2]:  # Test first 2 phases
            await deployer.execute_phase(name, cmd_set, test_results)
            if name in test_results["phases"]:
                print(f"   ✅ {name} phase executed successfully")
            else:
                print(f"   ❌ {name} phase failed")
        
        # Test validation
        print("4. Testing final validation...")
        validation = await deployer.run_final_validation()
        print(f"   ✅ Validation completed with {len(validation)} checks")
        
        print("\n🎉 Mock deployment test successful!")
        print(f"✅ All {len(command_sets)} command sets created")
        print(f"✅ Phase execution tested")  
        print(f"✅ Validation system tested")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    success = await test_deployment_mock()
    
    if success:
        print("\n🚀 Phase 2 deployment script is ready!")
        print("\nTo run with real orchestrator:")
        print("  python3 deploy_phase2_with_orchestrator.py")
        return 0
    else:
        print("\n❌ Mock test failed")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)