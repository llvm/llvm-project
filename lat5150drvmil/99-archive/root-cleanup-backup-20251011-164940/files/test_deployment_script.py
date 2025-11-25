#!/usr/bin/env python3
"""
Test runner for Phase 2 deployment script
Validates the script can import and initialize without executing full deployment
"""

import sys
import os
import asyncio
from pathlib import Path

# Add current directory to path
sys.path.insert(0, '/home/john/LAT5150DRVMIL')

async def test_script_import():
    """Test that the deployment script can be imported and initialized"""
    print("🧪 Testing Phase 2 Deployment Script")
    print("=" * 40)
    
    try:
        # Import the deployment script
        print("1. Testing import...")
        import deploy_phase2_with_orchestrator as deploy_script
        print("   ✅ Import successful")
        
        # Test orchestrator path
        print("2. Testing orchestrator path...")
        orchestrator_path = Path("/home/john/claude-backups/agents/src/python")
        if orchestrator_path.exists():
            print(f"   ✅ Orchestrator path exists: {orchestrator_path}")
        else:
            print(f"   ❌ Orchestrator path missing: {orchestrator_path}")
            return False
        
        # Test Phase2Deployer class
        print("3. Testing Phase2Deployer class...")
        deployer = deploy_script.Phase2Deployer()
        print("   ✅ Phase2Deployer instantiated")
        
        # Test configuration
        print("4. Testing configuration...")
        config = deploy_script.DEPLOYMENT_CONFIG
        print(f"   ✅ Config loaded - ID: {config['deployment_id'][:20]}...")
        
        # Test command set creation (without execution)
        print("5. Testing command set creation...")
        tpm_cmd = deployer.create_tpm_integration_commandset()
        ml_cmd = deployer.create_ml_system_commandset()
        device_cmd = deployer.create_device_activation_commandset()
        test_cmd = deployer.create_testing_commandset()
        doc_cmd = deployer.create_documentation_commandset()
        
        print(f"   ✅ TPM CommandSet: {len(tpm_cmd.steps)} steps")
        print(f"   ✅ ML CommandSet: {len(ml_cmd.steps)} steps")
        print(f"   ✅ Device CommandSet: {len(device_cmd.steps)} steps")
        print(f"   ✅ Testing CommandSet: {len(test_cmd.steps)} steps")
        print(f"   ✅ Documentation CommandSet: {len(doc_cmd.steps)} steps")
        
        # Test orchestrator initialization (but don't fully initialize)
        print("6. Testing orchestrator import...")
        from production_orchestrator import ProductionOrchestrator, ExecutionMode
        print("   ✅ Orchestrator classes imported successfully")
        
        print("\n✅ All tests passed! Deployment script is ready.")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

async def main():
    """Main test function"""
    success = await test_script_import()
    
    if success:
        print("\n🎯 Script validation complete - ready for deployment!")
        print("\nTo run actual deployment:")
        print("  python3 /home/john/LAT5150DRVMIL/deploy_phase2_with_orchestrator.py")
        return 0
    else:
        print("\n❌ Script validation failed - check errors above")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)