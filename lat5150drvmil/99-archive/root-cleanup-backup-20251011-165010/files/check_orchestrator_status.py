#!/usr/bin/env python3
"""
Quick utility to check Tandem Orchestrator status and available agents
"""

import sys
import asyncio
from pathlib import Path

# Add orchestrator to path and set agents root
orchestrator_path = "/home/john/claude-backups/agents/src/python"
sys.path.insert(0, orchestrator_path)
import os
os.environ['CLAUDE_AGENTS_ROOT'] = '/home/john/claude-backups/agents'

from production_orchestrator import ProductionOrchestrator

async def check_status():
    """Check orchestrator status and list agents"""
    print("🔍 Checking Tandem Orchestrator Status")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = ProductionOrchestrator()
    print("📡 Initializing orchestrator...")
    
    success = await orchestrator.initialize()
    if not success:
        print("❌ Failed to initialize orchestrator")
        return
    
    # Get status
    status = orchestrator.get_system_status()
    print(f"✅ Orchestrator initialized: {status['initialized']}")
    print(f"🕒 Uptime: {status['uptime_seconds']:.1f} seconds")
    print(f"🤖 Discovered agents: {status['discovered_agents']}")
    print(f"🔄 Active commands: {status['active_commands']}")
    print(f"🐍 Python messages: {status['python_msgs_processed']}")
    print(f"⚙️  C layer available: {status['c_layer_available']}")
    
    # Hardware topology
    hw = status['hardware_topology']
    print(f"\n🖥️  Hardware Topology:")
    print(f"   Total cores: {hw['total_cores']}")
    print(f"   P-cores (ultra): {hw['p_cores_ultra']}")
    print(f"   P-cores (standard): {hw['p_cores_standard']}")
    print(f"   E-cores: {hw['e_cores']}")
    print(f"   LP E-cores: {hw['lp_e_cores']}")
    
    # List all agents
    agents = orchestrator.get_agent_list()
    print(f"\n👥 Available Agents ({len(agents)}):")
    
    # Group agents by category (basic categorization)
    categories = {
        "Strategic": ["director", "projectorchestrator", "architect"],
        "Security": [a for a in agents if any(term in a for term in ['security', 'crypto', 'audit', 'ghost', 'apt'])],
        "Development": [a for a in agents if any(term in a for term in ['debug', 'test', 'patch', 'lint', 'construct'])],
        "Language": [a for a in agents if '-internal' in a or a in ['c', 'python', 'rust', 'go', 'java']],
        "Hardware": [a for a in agents if 'hardware' in a or a in ['npu', 'gna', 'leadengineer']],
        "Infrastructure": [a for a in agents if any(term in a for term in ['deploy', 'monitor', 'infra', 'docker', 'proxy'])],
        "Data/ML": [a for a in agents if any(term in a for term in ['data', 'ml', 'science'])],
        "Other": []
    }
    
    # Categorize remaining agents
    categorized = set()
    for agents_list in categories.values():
        categorized.update(agents_list)
    
    categories["Other"] = [a for a in agents if a not in categorized]
    
    for category, agent_list in categories.items():
        if agent_list:
            print(f"\n  📁 {category} ({len(agent_list)}):")
            for agent in sorted(agent_list):
                if agent in agents:  # Only show actually available agents
                    print(f"     • {agent}")
    
    print(f"\n🎯 Ready for Phase 2 deployment with {len(agents)} agents!")

if __name__ == "__main__":
    asyncio.run(check_status())