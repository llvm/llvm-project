#!/usr/bin/env python3
"""
ACE-FCA Subagent Registry Module
---------------------------------
Provides a registry pattern for subagent registration and discovery,
replacing the global factory dictionary with a more flexible, testable solution.

Addresses: Global state coupling, lack of dependency injection, testability issues
"""

from typing import Dict, Type, Optional, List, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SubagentCapability(Enum):
    """Capabilities that subagents can provide"""
    RESEARCH = "research"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class SubagentMetadata:
    """Metadata about a registered subagent"""
    agent_type: str
    agent_class: Type
    capabilities: List[SubagentCapability]
    description: str
    priority: int = 0  # Higher priority agents are preferred
    requires_human_review: bool = False


class SubagentRegistry:
    """
    Registry for subagent types with dependency injection support.

    Replaces the global _subagent_factory dictionary with a more flexible,
    testable solution that supports:
    - Dynamic registration
    - Discovery by capability
    - Dependency injection
    - Multiple registry instances (for testing)
    """

    def __init__(self):
        self._agents: Dict[str, SubagentMetadata] = {}
        self._capabilities: Dict[SubagentCapability, List[str]] = {}
        self._factory_overrides: Dict[str, Callable] = {}

    def register(
        self,
        agent_type: str,
        agent_class: Type,
        capabilities: List[SubagentCapability],
        description: str = "",
        priority: int = 0,
        requires_human_review: bool = False
    ):
        """
        Register a subagent type.

        Args:
            agent_type: Unique identifier for the agent
            agent_class: The agent class to instantiate
            capabilities: List of capabilities the agent provides
            description: Human-readable description
            priority: Priority (higher = preferred when multiple agents match)
            requires_human_review: Whether the agent requires human review
        """
        metadata = SubagentMetadata(
            agent_type=agent_type,
            agent_class=agent_class,
            capabilities=capabilities,
            description=description,
            priority=priority,
            requires_human_review=requires_human_review
        )

        self._agents[agent_type] = metadata

        # Index by capabilities
        for capability in capabilities:
            if capability not in self._capabilities:
                self._capabilities[capability] = []
            self._capabilities[capability].append(agent_type)

        logger.info(f"Registered subagent: {agent_type} with capabilities {[c.value for c in capabilities]}")

    def unregister(self, agent_type: str):
        """Unregister a subagent type"""
        if agent_type not in self._agents:
            return

        metadata = self._agents[agent_type]

        # Remove from capability index
        for capability in metadata.capabilities:
            if capability in self._capabilities:
                self._capabilities[capability].remove(agent_type)

        del self._agents[agent_type]
        logger.info(f"Unregistered subagent: {agent_type}")

    def get_metadata(self, agent_type: str) -> Optional[SubagentMetadata]:
        """Get metadata for a registered subagent"""
        return self._agents.get(agent_type)

    def create_agent(
        self,
        agent_type: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Create an instance of a registered subagent.

        Args:
            agent_type: Type of agent to create
            *args, **kwargs: Arguments to pass to agent constructor

        Returns:
            Instance of the agent

        Raises:
            ValueError: If agent type is not registered
        """
        # Check for factory override (for testing)
        if agent_type in self._factory_overrides:
            return self._factory_overrides[agent_type](*args, **kwargs)

        metadata = self._agents.get(agent_type)
        if not metadata:
            available = ", ".join(self._agents.keys())
            raise ValueError(
                f"Unknown subagent type: {agent_type}. "
                f"Available types: {available}"
            )

        try:
            return metadata.agent_class(*args, **kwargs)
        except Exception as e:
            logger.error(f"Failed to create subagent {agent_type}: {e}")
            raise

    def find_by_capability(
        self,
        capability: SubagentCapability,
        sort_by_priority: bool = True
    ) -> List[str]:
        """
        Find all agent types that provide a specific capability.

        Args:
            capability: The capability to search for
            sort_by_priority: Whether to sort by priority (highest first)

        Returns:
            List of agent type names
        """
        agent_types = self._capabilities.get(capability, [])

        if sort_by_priority:
            # Sort by priority (descending)
            agent_types = sorted(
                agent_types,
                key=lambda t: self._agents[t].priority,
                reverse=True
            )

        return agent_types

    def get_all_types(self) -> List[str]:
        """Get all registered agent types"""
        return list(self._agents.keys())

    def get_all_capabilities(self) -> List[SubagentCapability]:
        """Get all capabilities provided by registered agents"""
        return list(self._capabilities.keys())

    def set_factory_override(self, agent_type: str, factory: Callable):
        """
        Override the factory for a specific agent type (for testing).

        Args:
            agent_type: Agent type to override
            factory: Factory function that creates the agent
        """
        self._factory_overrides[agent_type] = factory

    def clear_factory_overrides(self):
        """Clear all factory overrides"""
        self._factory_overrides.clear()

    def is_registered(self, agent_type: str) -> bool:
        """Check if an agent type is registered"""
        return agent_type in self._agents

    def get_agent_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered agents.

        Returns:
            Dictionary mapping agent types to their info
        """
        return {
            agent_type: {
                'description': metadata.description,
                'capabilities': [c.value for c in metadata.capabilities],
                'priority': metadata.priority,
                'requires_review': metadata.requires_human_review,
                'class': metadata.agent_class.__name__
            }
            for agent_type, metadata in self._agents.items()
        }


# =============================================================================
# Decorator for Easy Registration
# =============================================================================

def subagent(
    agent_type: str,
    capabilities: List[SubagentCapability],
    description: str = "",
    priority: int = 0,
    requires_human_review: bool = False,
    registry: Optional[SubagentRegistry] = None
):
    """
    Decorator for registering subagent classes.

    Usage:
        @subagent(
            agent_type="research",
            capabilities=[SubagentCapability.RESEARCH],
            description="Research agent for code analysis"
        )
        class ResearchAgent(BaseSubagent):
            pass
    """
    def decorator(cls):
        target_registry = registry or get_default_registry()
        target_registry.register(
            agent_type=agent_type,
            agent_class=cls,
            capabilities=capabilities,
            description=description,
            priority=priority,
            requires_human_review=requires_human_review
        )
        return cls
    return decorator


# =============================================================================
# Global Default Registry
# =============================================================================

_default_registry: Optional[SubagentRegistry] = None


def get_default_registry() -> SubagentRegistry:
    """Get the default global registry (lazy initialization)"""
    global _default_registry
    if _default_registry is None:
        _default_registry = SubagentRegistry()
    return _default_registry


def set_default_registry(registry: SubagentRegistry):
    """Set the default global registry (useful for testing)"""
    global _default_registry
    _default_registry = registry


def reset_default_registry():
    """Reset the default registry to a new empty instance"""
    global _default_registry
    _default_registry = SubagentRegistry()


# =============================================================================
# Convenience Functions
# =============================================================================

def register_subagent(
    agent_type: str,
    agent_class: Type,
    capabilities: List[SubagentCapability],
    description: str = "",
    priority: int = 0,
    requires_human_review: bool = False
):
    """Register a subagent in the default registry"""
    registry = get_default_registry()
    registry.register(
        agent_type=agent_type,
        agent_class=agent_class,
        capabilities=capabilities,
        description=description,
        priority=priority,
        requires_human_review=requires_human_review
    )


def create_subagent(agent_type: str, *args, **kwargs) -> Any:
    """Create a subagent from the default registry"""
    registry = get_default_registry()
    return registry.create_agent(agent_type, *args, **kwargs)


def find_subagents_by_capability(
    capability: SubagentCapability,
    sort_by_priority: bool = True
) -> List[str]:
    """Find subagents by capability in the default registry"""
    registry = get_default_registry()
    return registry.find_by_capability(capability, sort_by_priority)


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    # Create a test registry
    registry = SubagentRegistry()

    # Mock agent class
    class MockResearchAgent:
        def __init__(self, ai_engine=None):
            self.ai_engine = ai_engine

    class MockPlanningAgent:
        def __init__(self, ai_engine=None):
            self.ai_engine = ai_engine

    # Register agents
    registry.register(
        agent_type="research",
        agent_class=MockResearchAgent,
        capabilities=[SubagentCapability.RESEARCH, SubagentCapability.CODE_GENERATION],
        description="Research agent for code analysis",
        priority=10
    )

    registry.register(
        agent_type="planning",
        agent_class=MockPlanningAgent,
        capabilities=[SubagentCapability.PLANNING],
        description="Planning agent for task breakdown",
        priority=5
    )

    # Test discovery
    print("All registered agents:", registry.get_all_types())
    print("\nResearch capability:", registry.find_by_capability(SubagentCapability.RESEARCH))
    print("Planning capability:", registry.find_by_capability(SubagentCapability.PLANNING))

    # Test creation
    research_agent = registry.create_agent("research")
    print(f"\nCreated agent: {research_agent.__class__.__name__}")

    # Test info
    import json
    print("\nAgent info:")
    print(json.dumps(registry.get_agent_info(), indent=2))

    # Test decorator
    @subagent(
        agent_type="implementation",
        capabilities=[SubagentCapability.IMPLEMENTATION, SubagentCapability.CODE_GENERATION],
        description="Implementation agent",
        registry=registry
    )
    class MockImplementationAgent:
        pass

    print("\nAfter decorator registration:", registry.get_all_types())

    # Test factory override (for testing)
    mock_instance = MockResearchAgent()
    registry.set_factory_override("research", lambda *args, **kwargs: mock_instance)
    agent = registry.create_agent("research")
    print(f"\nFactory override works: {agent is mock_instance}")
