#!/usr/bin/env python3
"""
Local Agent Loader - Import and Convert claude-backups Agents
Standardizes agent definitions for local-only execution

Features:
- Imports agent definitions from claude-backups markdown files
- Converts to standardized Python classes
- Integrates with heterogeneous executor for hardware routing
- LOCAL-ONLY execution (no cloud dependencies)
- Compatible with comprehensive_98_agent_system.py
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AgentCategory(Enum):
    """Standardized agent categories"""
    STRATEGIC = "strategic"
    DEVELOPMENT = "development"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    QA = "qa"
    DOCUMENTATION = "documentation"
    OPERATIONS = "operations"
    SPECIALIZED = "specialized"  # For agents that don't fit other categories


class ExecutionMode(Enum):
    """How the agent executes"""
    LOCAL_ONLY = "local_only"          # Never uses external APIs
    LOCAL_PREFERRED = "local_preferred" # Prefers local, fallback to online
    HYBRID = "hybrid"                  # Uses both local and online strategically


@dataclass
class StandardizedAgent:
    """
    Standardized agent definition for local execution
    Converted from claude-backups markdown format
    """
    id: str                              # Unique identifier
    name: str                            # Display name
    category: AgentCategory              # Category for organization
    role: str                            # Primary role description
    capabilities: List[str] = field(default_factory=list)  # What it can do
    execution_mode: ExecutionMode = ExecutionMode.LOCAL_ONLY

    # Hardware preferences (integrated with heterogeneous executor)
    preferred_hardware: str = "CPU"      # NPU, GNA, CPU_PCORE, CPU_ECORE, CPU
    requires_avx512: bool = False
    requires_npu: bool = False
    requires_gpu: bool = False

    # Local model configuration
    local_model: Optional[str] = None    # e.g., "wizardlm-uncensored-codellama:34b"
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Tools and dependencies (local alternatives only)
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)

    # Agent delegation (only to other local agents)
    can_delegate_to: List[str] = field(default_factory=list)

    # Performance characteristics
    avg_response_time_ms: float = 1000
    parallel_capable: bool = False

    # Original metadata from claude-backups
    original_version: str = "unknown"
    source_file: Optional[str] = None


class AgentMarkdownParser:
    """
    Parse claude-backups markdown agent definitions and convert to standardized format
    """

    def __init__(self):
        self.category_mappings = {
            # Strategic agents
            "ARCHITECT": AgentCategory.STRATEGIC,
            "PLANNER": AgentCategory.STRATEGIC,
            "COORDINATOR": AgentCategory.STRATEGIC,
            "DIRECTOR": AgentCategory.STRATEGIC,
            "LEADENGINEER": AgentCategory.STRATEGIC,

            # Development agents
            "PYTHON-INTERNAL": AgentCategory.DEVELOPMENT,
            "RUST-INTERNAL-AGENT": AgentCategory.DEVELOPMENT,
            "CPP-INTERNAL-AGENT": AgentCategory.DEVELOPMENT,
            "GO-INTERNAL-AGENT": AgentCategory.DEVELOPMENT,
            "JAVA-INTERNAL": AgentCategory.DEVELOPMENT,
            "TYPESCRIPT-INTERNAL-AGENT": AgentCategory.DEVELOPMENT,
            "DEBUGGER": AgentCategory.DEVELOPMENT,
            "LINTER": AgentCategory.DEVELOPMENT,
            "OPTIMIZER": AgentCategory.DEVELOPMENT,

            # Infrastructure agents
            "DOCKER-AGENT": AgentCategory.INFRASTRUCTURE,
            "PROXMOX-AGENT": AgentCategory.INFRASTRUCTURE,
            "ZFS-INTERNAL": AgentCategory.INFRASTRUCTURE,
            "DEPLOYER": AgentCategory.INFRASTRUCTURE,
            "PATCHER": AgentCategory.INFRASTRUCTURE,
            "CONSTRUCTOR": AgentCategory.INFRASTRUCTURE,

            # Security agents
            "SECURITY": AgentCategory.SECURITY,
            "SECURITYAUDITOR": AgentCategory.SECURITY,
            "BASTION": AgentCategory.SECURITY,
            "COGNITIVE_DEFENSE_AGENT": AgentCategory.SECURITY,
            "RED-TEAM": AgentCategory.SECURITY,
            "BLUE-TEAM": AgentCategory.SECURITY,

            # QA agents
            "TESTBED": AgentCategory.QA,
            "MONITOR": AgentCategory.QA,
            "AUDITOR": AgentCategory.QA,

            # Documentation agents
            "DOCGEN": AgentCategory.DOCUMENTATION,
            "RESEARCHER": AgentCategory.DOCUMENTATION,

            # Operations agents
            "ORCHESTRATOR": AgentCategory.OPERATIONS,
            "PROJECTORCHESTRATOR": AgentCategory.OPERATIONS,
        }

    def parse_markdown_agent(self, content: str, filename: str) -> StandardizedAgent:
        """
        Parse a markdown agent definition from claude-backups

        Args:
            content: Markdown file content
            filename: Original filename for ID

        Returns:
            Standardized agent definition
        """
        # Extract agent ID from filename
        agent_id = filename.replace(".md", "").lower().replace("-", "_")

        # Extract name (first heading or from filename)
        name_match = re.search(r'^#\s+(.+?)(?:\s+v[\d.]+)?', content, re.MULTILINE)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = filename.replace(".md", "").replace("-", " ").title()

        # Extract version
        version_match = re.search(r'v([\d.]+)', content)
        version = version_match.group(1) if version_match else "1.0"

        # Extract role/description
        role = self._extract_role(content)

        # Extract capabilities
        capabilities = self._extract_capabilities(content)

        # Determine category
        category = self._determine_category(agent_id, content)

        # Determine execution mode (force local-only for our system)
        execution_mode = ExecutionMode.LOCAL_ONLY

        # Hardware preferences based on agent type
        preferred_hardware, requires_avx512, requires_npu = self._determine_hardware_prefs(agent_id, content, category)

        # Local model selection
        local_model = self._select_local_model(category, capabilities)

        # Tools (convert to local equivalents)
        required_tools, optional_tools = self._extract_tools(content)

        # Delegation (only local agents)
        can_delegate_to = self._extract_delegation(content)

        return StandardizedAgent(
            id=agent_id,
            name=name,
            category=category,
            role=role,
            capabilities=capabilities,
            execution_mode=execution_mode,
            preferred_hardware=preferred_hardware,
            requires_avx512=requires_avx512,
            requires_npu=requires_npu,
            local_model=local_model,
            model_params={"temperature": 0.3, "max_tokens": 4096},
            required_tools=required_tools,
            optional_tools=optional_tools,
            can_delegate_to=can_delegate_to,
            original_version=version,
            source_file=filename
        )

    def _extract_role(self, content: str) -> str:
        """Extract role description from markdown"""
        # Look for "Role:", "Core Identity", or similar
        role_patterns = [
            r'##\s*Core Identity & Role\s*\n(.+?)(?:\n##|$)',
            r'##\s*Role\s*\n(.+?)(?:\n##|$)',
            r'##\s*Primary Role\s*\n(.+?)(?:\n##|$)',
            r'"(.+?)"',  # Quoted description
        ]

        for pattern in role_patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                role = match.group(1).strip()
                # Clean up
                role = re.sub(r'\n+', ' ', role)
                role = re.sub(r'\s+', ' ', role)
                return role[:500]  # Limit length

        return "Specialized agent for task execution"

    def _extract_capabilities(self, content: str) -> List[str]:
        """Extract capabilities list from markdown"""
        capabilities = []

        # Look for capabilities section
        cap_match = re.search(r'##\s*(?:Primary\s+)?Capabilities?\s*\n(.+?)(?:\n##|$)', content, re.DOTALL | re.IGNORECASE)
        if cap_match:
            cap_text = cap_match.group(1)

            # Extract bullet points or numbered items
            for line in cap_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line):
                    # Remove bullet/number
                    cap = re.sub(r'^[-*\d.]\s*', '', line).strip()
                    # Remove markdown formatting
                    cap = re.sub(r'\*\*(.+?)\*\*', r'\1', cap)
                    if cap and len(cap) > 5:
                        capabilities.append(cap)

        # If no structured capabilities, extract from content
        if not capabilities:
            # Look for action verbs
            action_patterns = [
                r'can\s+([a-z][^.!?]+)',
                r'supports?\s+([a-z][^.!?]+)',
                r'provides?\s+([a-z][^.!?]+)',
                r'enables?\s+([a-z][^.!?]+)',
            ]

            for pattern in action_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    cap = match.group(1).strip()
                    if len(cap) > 10 and len(cap) < 200:
                        capabilities.append(cap)
                        if len(capabilities) >= 10:
                            break
                if len(capabilities) >= 10:
                    break

        return capabilities[:10]  # Limit to 10 capabilities

    def _determine_category(self, agent_id: str, content: str) -> AgentCategory:
        """Determine agent category from ID and content"""
        agent_id_upper = agent_id.upper().replace("_", "-")

        # Check predefined mappings
        for key, category in self.category_mappings.items():
            if key in agent_id_upper:
                return category

        # Infer from content
        content_lower = content.lower()

        if any(word in content_lower for word in ['security', 'audit', 'threat', 'vulnerability', 'exploit']):
            return AgentCategory.SECURITY
        elif any(word in content_lower for word in ['test', 'qa', 'validation', 'coverage']):
            return AgentCategory.QA
        elif any(word in content_lower for word in ['document', 'research', 'analysis']):
            return AgentCategory.DOCUMENTATION
        elif any(word in content_lower for word in ['deploy', 'orchestrat', 'monitor', 'incident']):
            return AgentCategory.OPERATIONS
        elif any(word in content_lower for word in ['infrastructure', 'docker', 'kubernetes', 'cloud']):
            return AgentCategory.INFRASTRUCTURE
        elif any(word in content_lower for word in ['python', 'rust', 'java', 'code', 'programming']):
            return AgentCategory.DEVELOPMENT
        elif any(word in content_lower for word in ['architect', 'plan', 'strategy', 'design']):
            return AgentCategory.STRATEGIC
        else:
            return AgentCategory.SPECIALIZED

    def _determine_hardware_prefs(self, agent_id: str, content: str, category: AgentCategory) -> tuple:
        """Determine hardware preferences based on agent type"""
        content_lower = content.lower()

        # NPU for real-time inference, audio processing
        if any(word in content_lower for word in ['real-time', 'inference', 'audio', 'voice', 'speech']):
            return ("NPU", False, True)

        # GNA for continuous low-power operations
        if 'continuous' in content_lower or 'low-power' in content_lower or 'wake word' in content_lower:
            return ("GNA", False, False)

        # P-cores for complex reasoning, compilation, optimization
        if category in [AgentCategory.STRATEGIC, AgentCategory.DEVELOPMENT]:
            if any(word in content_lower for word in ['complex', 'reasoning', 'optimization', 'compilation']):
                return ("CPU_PCORE", True, False)

        # AVX512 for data processing, crypto, compression
        if any(word in content_lower for word in ['crypto', 'compression', 'parallel', 'vector']):
            return ("CPU_PCORE", True, False)

        # E-cores for background tasks
        if any(word in content_lower for word in ['background', 'monitoring', 'logging']):
            return ("CPU_ECORE", False, False)

        # Default: standard CPU
        return ("CPU", False, False)

    def _select_local_model(self, category: AgentCategory, capabilities: List[str]) -> str:
        """Select appropriate local model based on agent requirements"""
        # Map categories to local models
        model_mapping = {
            AgentCategory.STRATEGIC: "wizardlm-uncensored-codellama:34b",  # Complex reasoning
            AgentCategory.DEVELOPMENT: "deepseek-coder:6.7b",               # Code generation
            AgentCategory.INFRASTRUCTURE: "deepseek-coder:6.7b",            # Technical tasks
            AgentCategory.SECURITY: "wizardlm-uncensored-codellama:34b",   # Security analysis
            AgentCategory.QA: "qwen2.5-coder:14b",                          # Testing and validation
            AgentCategory.DOCUMENTATION: "qwen2.5-coder:14b",               # Technical writing
            AgentCategory.OPERATIONS: "wizardlm-uncensored-codellama:34b", # Operational decisions
            AgentCategory.SPECIALIZED: "deepseek-coder:6.7b",               # General purpose
        }

        return model_mapping.get(category, "deepseek-coder:6.7b")

    def _extract_tools(self, content: str) -> tuple:
        """Extract required and optional tools (convert to local equivalents)"""
        required = []
        optional = []

        # Look for tools section
        tools_match = re.search(r'##\s*(?:Required\s+)?Tools?\s*\n(.+?)(?:\n##|$)', content, re.DOTALL | re.IGNORECASE)
        if tools_match:
            tools_text = tools_match.group(1)

            for line in tools_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*'):
                    tool = re.sub(r'^[-*]\s*\*?\*?', '', line).strip()
                    tool = re.sub(r'\*\*(.+?)\*\*', r'\1', tool)

                    # Convert to local equivalents
                    tool_local = self._convert_to_local_tool(tool)
                    if tool_local:
                        if 'mandatory' in line.lower() or 'required' in line.lower():
                            required.append(tool_local)
                        else:
                            optional.append(tool_local)

        # Local-only tools we always support
        base_tools = ["filesystem_read", "filesystem_write", "bash_execute", "grep_search"]
        required.extend([t for t in base_tools if t not in required])

        return required, optional

    def _convert_to_local_tool(self, tool: str) -> Optional[str]:
        """Convert cloud-based tools to local equivalents"""
        tool_lower = tool.lower()

        # Local file operations
        if any(word in tool_lower for word in ['read', 'write', 'edit']):
            return "filesystem_operations"

        # Local code execution
        if any(word in tool_lower for word in ['bash', 'execute', 'run']):
            return "bash_execute"

        # Local search
        if any(word in tool_lower for word in ['grep', 'search', 'find']):
            return "grep_search"

        # Skip cloud-only tools
        if any(word in tool_lower for word in ['web', 'api', 'cloud', 'remote']):
            return None

        return tool.lower().replace(' ', '_')

    def _extract_delegation(self, content: str) -> List[str]:
        """Extract which other agents this agent can delegate to"""
        delegates = []

        # Look for delegation patterns
        patterns = [
            r'delegates?\s+to:?\s*(.+)',
            r'invokes?:?\s*(.+)',
            r'coordinates?\s+with:?\s*(.+)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                delegate_text = match.group(1)
                # Extract agent names (typically capitalized or in code blocks)
                agent_names = re.findall(r'[A-Z][A-Z_-]+(?:-AGENT)?', delegate_text)
                delegates.extend([name.lower().replace('-', '_') for name in agent_names])

        return list(set(delegates))[:10]  # Limit and deduplicate


class LocalAgentLoader:
    """
    Load and manage agents from claude-backups repository
    Converts to standardized format for local execution
    """

    def __init__(self, agents_source_dir: Optional[str] = None):
        """
        Initialize agent loader

        Args:
            agents_source_dir: Path to claude-backups/agents directory (optional)
        """
        self.parser = AgentMarkdownParser()
        self.agents: Dict[str, StandardizedAgent] = {}
        self.agents_source_dir = agents_source_dir

        print("✓ Local Agent Loader initialized")

    def load_from_directory(self, directory: str) -> int:
        """
        Load all agent definitions from a directory

        Args:
            directory: Path to directory containing .md agent files

        Returns:
            Number of agents loaded
        """
        directory_path = Path(directory)

        if not directory_path.exists():
            print(f"⚠️  Directory not found: {directory}")
            return 0

        print(f"Loading agents from: {directory}")

        count = 0
        for md_file in directory_path.glob("*.md"):
            if md_file.name.startswith('.') or md_file.name.lower() == 'readme.md':
                continue

            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                agent = self.parser.parse_markdown_agent(content, md_file.name)
                self.agents[agent.id] = agent
                count += 1

                if count % 10 == 0:
                    print(f"  Loaded {count} agents...")

            except Exception as e:
                print(f"  ⚠️  Failed to load {md_file.name}: {e}")

        print(f"✓ Loaded {count} agents")
        return count

    def load_agent_definition(self, agent_id: str, markdown_content: str) -> StandardizedAgent:
        """
        Load a single agent from markdown content

        Args:
            agent_id: Agent identifier
            markdown_content: Markdown content

        Returns:
            Standardized agent
        """
        agent = self.parser.parse_markdown_agent(markdown_content, f"{agent_id}.md")
        self.agents[agent_id] = agent
        return agent

    def get_agent(self, agent_id: str) -> Optional[StandardizedAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)

    def get_agents_by_category(self, category: AgentCategory) -> List[StandardizedAgent]:
        """Get all agents in a category"""
        return [agent for agent in self.agents.values() if agent.category == category]

    def get_agents_for_hardware(self, hardware: str) -> List[StandardizedAgent]:
        """Get agents optimized for specific hardware"""
        return [agent for agent in self.agents.values() if agent.preferred_hardware == hardware]

    def export_to_json(self, output_file: str):
        """Export loaded agents to JSON"""
        agents_dict = {}
        for agent_id, agent in self.agents.items():
            agents_dict[agent_id] = {
                "id": agent.id,
                "name": agent.name,
                "category": agent.category.value,
                "role": agent.role,
                "capabilities": agent.capabilities,
                "execution_mode": agent.execution_mode.value,
                "preferred_hardware": agent.preferred_hardware,
                "local_model": agent.local_model,
                "required_tools": agent.required_tools,
                "can_delegate_to": agent.can_delegate_to,
            }

        with open(output_file, 'w') as f:
            json.dump(agents_dict, f, indent=2)

        print(f"✓ Exported {len(agents_dict)} agents to {output_file}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded agents"""
        stats = {
            "total": len(self.agents),
            "by_category": {},
            "by_hardware": {},
            "by_execution_mode": {},
            "requires_npu": 0,
            "requires_avx512": 0,
        }

        for agent in self.agents.values():
            # By category
            cat = agent.category.value
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # By hardware
            hw = agent.preferred_hardware
            stats["by_hardware"][hw] = stats["by_hardware"].get(hw, 0) + 1

            # By execution mode
            mode = agent.execution_mode.value
            stats["by_execution_mode"][mode] = stats["by_execution_mode"].get(mode, 0) + 1

            # Hardware requirements
            if agent.requires_npu:
                stats["requires_npu"] += 1
            if agent.requires_avx512:
                stats["requires_avx512"] += 1

        return stats


def demo():
    """Demonstration of local agent loading"""
    print("=" * 70)
    print(" Local Agent Loader Demo")
    print("=" * 70)
    print()

    loader = LocalAgentLoader()

    # Example: Load a single agent definition
    python_agent_md = """
# PYTHON-INTERNAL v8.0

## Core Identity & Role
Elite Python execution specialist with advanced parallel processing capabilities

## Primary Capabilities
- Standard Python execution (python 3.11+)
- Parallel processing across 22 cores
- Hardware acceleration: AVX-512, NPU integration
- Code formatting (Black, isort)
- Linting with Ruff and MyPy
- Security scanning via Bandit
    """

    agent = loader.load_agent_definition("python_internal", python_agent_md)

    print(f"Loaded Agent: {agent.name}")
    print(f"  Category: {agent.category.value}")
    print(f"  Role: {agent.role[:100]}...")
    print(f"  Capabilities: {len(agent.capabilities)}")
    print(f"  Hardware: {agent.preferred_hardware}")
    print(f"  Local Model: {agent.local_model}")
    print(f"  Execution Mode: {agent.execution_mode.value}")
    print()

    # Stats
    stats = loader.get_stats()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
