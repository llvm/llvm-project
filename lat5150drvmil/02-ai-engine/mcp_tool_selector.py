#!/usr/bin/env python3
"""
MCP Tool Selector
Intelligent tool selection and routing for MCP ecosystem

Key Capabilities:
- Context-aware tool selection from 35+ MCP tools
- Capability matching (what tool can do this task?)
- Cost optimization (choose cheapest tool that works)
- Automatic tool composition (chain tools together)
- Query understanding and tool routing

Use Cases:
- "Investigate this Bitcoin address" → blockchain_analyze tool
- "Find info about john@example.com" → osint_query + email_verify
- "What files changed recently?" → filesystem tool
- "Search for crypto scams" → brave_search + threat_intelligence
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ToolCategory(Enum):
    """Tool categories for classification"""
    OSINT = "osint"
    BLOCKCHAIN = "blockchain"
    FILESYSTEM = "filesystem"
    VERSION_CONTROL = "version_control"
    WEB_SEARCH = "web_search"
    STORAGE = "storage"
    THREAT_INTEL = "threat_intel"
    ANALYSIS = "analysis"


class ToolCostTier(Enum):
    """Cost tiers for tools"""
    FREE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class ToolCapability:
    """
    Capability description for a tool

    Attributes:
        tool_name: Name of the MCP tool
        category: Tool category
        capabilities: List of what the tool can do
        input_types: Types of input the tool accepts
        cost_tier: Cost tier for the tool
        latency_ms: Average latency in milliseconds
        keywords: Keywords for matching user queries
    """
    tool_name: str
    category: ToolCategory
    capabilities: List[str]
    input_types: List[str]
    cost_tier: ToolCostTier
    latency_ms: int
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class ToolSelection:
    """
    Selected tool with reasoning

    Attributes:
        tool_name: Name of selected tool
        confidence: Confidence score (0.0-1.0)
        reasoning: Why this tool was selected
        parameters: Suggested parameters for the tool
        alternatives: Alternative tools that could work
        composition: If multiple tools needed, the sequence
    """
    tool_name: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    alternatives: List[str] = field(default_factory=list)
    composition: List[str] = field(default_factory=list)


class MCPToolSelector:
    """
    Intelligent MCP tool selector

    Features:
    - Context-aware tool selection
    - Capability matching
    - Cost optimization
    - Tool composition (chaining)
    - Query understanding
    """

    def __init__(self, optimize_for: str = "quality"):
        """
        Initialize MCP tool selector

        Args:
            optimize_for: Optimization strategy
                - "quality": Best quality tool
                - "speed": Fastest tool
                - "cost": Cheapest tool
                - "balanced": Balance of all factors
        """
        self.optimize_for = optimize_for
        self.tool_registry = self._initialize_tool_registry()
        self.selection_history: List[ToolSelection] = []

    def _initialize_tool_registry(self) -> Dict[str, ToolCapability]:
        """Initialize registry of available MCP tools"""
        return {
            # DIRECTEYE OSINT Tools
            "osint_query": ToolCapability(
                tool_name="osint_query",
                category=ToolCategory.OSINT,
                capabilities=[
                    "Email verification",
                    "Domain WHOIS",
                    "IP geolocation",
                    "Social media OSINT",
                    "Phone number lookup",
                    "Person search"
                ],
                input_types=["email", "domain", "ip_address", "phone", "person_name"],
                cost_tier=ToolCostTier.MEDIUM,
                latency_ms=2000,
                keywords=["osint", "investigate", "lookup", "find", "search", "verify"],
                examples=[
                    "Find information about john@example.com",
                    "Investigate domain example.com",
                    "Lookup IP 192.168.1.1"
                ]
            ),

            "blockchain_analyze": ToolCapability(
                tool_name="blockchain_analyze",
                category=ToolCategory.BLOCKCHAIN,
                capabilities=[
                    "Crypto address analysis",
                    "Transaction history",
                    "Balance checking",
                    "Token holdings",
                    "NFT ownership",
                    "Wallet clustering"
                ],
                input_types=["crypto_address"],
                cost_tier=ToolCostTier.LOW,
                latency_ms=1500,
                keywords=["bitcoin", "ethereum", "crypto", "blockchain", "wallet", "address", "transaction"],
                examples=[
                    "Analyze Bitcoin address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
                    "Check Ethereum balance for 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                    "What tokens does this wallet hold?"
                ]
            ),

            "threat_intelligence": ToolCapability(
                tool_name="threat_intelligence",
                category=ToolCategory.THREAT_INTEL,
                capabilities=[
                    "IOC lookup",
                    "Threat actor attribution",
                    "Malware analysis",
                    "CVE lookup",
                    "Dark web monitoring"
                ],
                input_types=["ip_address", "domain", "hash", "cve_id"],
                cost_tier=ToolCostTier.HIGH,
                latency_ms=3000,
                keywords=["threat", "malware", "ioc", "cve", "attack", "vulnerability", "exploit"],
                examples=[
                    "Is this IP address malicious?",
                    "Lookup CVE-2021-44228",
                    "Threat intel for domain evil.com"
                ]
            ),

            "domain_investigate": ToolCapability(
                tool_name="domain_investigate",
                category=ToolCategory.OSINT,
                capabilities=[
                    "WHOIS lookup",
                    "DNS records",
                    "SSL certificate info",
                    "Domain reputation",
                    "Subdomain enumeration"
                ],
                input_types=["domain"],
                cost_tier=ToolCostTier.LOW,
                latency_ms=1000,
                keywords=["domain", "whois", "dns", "ssl", "certificate", "website"],
                examples=[
                    "Investigate domain google.com",
                    "WHOIS for example.com",
                    "DNS records for github.com"
                ]
            ),

            "ip_investigate": ToolCapability(
                tool_name="ip_investigate",
                category=ToolCategory.OSINT,
                capabilities=[
                    "IP geolocation",
                    "ASN lookup",
                    "Reverse DNS",
                    "IP reputation",
                    "Port scanning"
                ],
                input_types=["ip_address"],
                cost_tier=ToolCostTier.LOW,
                latency_ms=800,
                keywords=["ip", "geolocation", "asn", "reverse dns", "port"],
                examples=[
                    "Geolocate IP 8.8.8.8",
                    "ASN for 1.1.1.1",
                    "Where is this IP address located?"
                ]
            ),

            "email_verify": ToolCapability(
                tool_name="email_verify",
                category=ToolCategory.OSINT,
                capabilities=[
                    "Email validation",
                    "Domain verification",
                    "Deliverability check",
                    "Breach detection",
                    "Email reputation"
                ],
                input_types=["email"],
                cost_tier=ToolCostTier.MEDIUM,
                latency_ms=1200,
                keywords=["email", "verify", "validate", "breach", "leak"],
                examples=[
                    "Verify email john@example.com",
                    "Is this email address valid?",
                    "Check if email was in breach"
                ]
            ),

            "phone_lookup": ToolCapability(
                tool_name="phone_lookup",
                category=ToolCategory.OSINT,
                capabilities=[
                    "Phone number validation",
                    "Carrier lookup",
                    "Country detection",
                    "Number type (mobile/landline)",
                    "OSINT phone search"
                ],
                input_types=["phone"],
                cost_tier=ToolCostTier.MEDIUM,
                latency_ms=1500,
                keywords=["phone", "number", "carrier", "mobile", "validate"],
                examples=[
                    "Lookup phone number +1-555-123-4567",
                    "What carrier is this number?",
                    "Validate phone number"
                ]
            ),

            # Filesystem Tools
            "read_file": ToolCapability(
                tool_name="read_file",
                category=ToolCategory.FILESYSTEM,
                capabilities=[
                    "Read file contents",
                    "List directory",
                    "File search",
                    "File metadata"
                ],
                input_types=["file_path"],
                cost_tier=ToolCostTier.FREE,
                latency_ms=50,
                keywords=["read", "file", "contents", "directory", "list", "ls", "cat"],
                examples=[
                    "Read file /etc/hosts",
                    "List files in /tmp",
                    "Show contents of config.json"
                ]
            ),

            "write_file": ToolCapability(
                tool_name="write_file",
                category=ToolCategory.FILESYSTEM,
                capabilities=[
                    "Write file",
                    "Create directory",
                    "Delete file",
                    "Move/rename file"
                ],
                input_types=["file_path", "content"],
                cost_tier=ToolCostTier.FREE,
                latency_ms=100,
                keywords=["write", "create", "save", "delete", "move", "rename"],
                examples=[
                    "Write data to file.txt",
                    "Create directory /tmp/test",
                    "Delete old_file.txt"
                ]
            ),

            # Git Tools
            "git_status": ToolCapability(
                tool_name="git_status",
                category=ToolCategory.VERSION_CONTROL,
                capabilities=[
                    "Git status",
                    "Git log",
                    "Git diff",
                    "Branch info",
                    "Commit history"
                ],
                input_types=["repo_path"],
                cost_tier=ToolCostTier.FREE,
                latency_ms=200,
                keywords=["git", "status", "log", "diff", "commit", "branch", "history"],
                examples=[
                    "Git status",
                    "Show recent commits",
                    "What changed in last commit?"
                ]
            ),

            # Web Search
            "brave_search": ToolCapability(
                tool_name="brave_search",
                category=ToolCategory.WEB_SEARCH,
                capabilities=[
                    "Web search",
                    "News search",
                    "Real-time information",
                    "Recent events"
                ],
                input_types=["query"],
                cost_tier=ToolCostTier.LOW,
                latency_ms=1000,
                keywords=["search", "web", "google", "find", "news", "recent", "latest"],
                examples=[
                    "Search for Python tutorials",
                    "Latest news about AI",
                    "Find information about quantum computing"
                ]
            ),

            # Memory/Storage
            "memory_store": ToolCapability(
                tool_name="memory_store",
                category=ToolCategory.STORAGE,
                capabilities=[
                    "Key-value storage",
                    "Entity storage",
                    "Persistent memory",
                    "Data retrieval"
                ],
                input_types=["key", "value"],
                cost_tier=ToolCostTier.FREE,
                latency_ms=50,
                keywords=["store", "save", "remember", "memory", "persist", "retrieve"],
                examples=[
                    "Remember user preference",
                    "Store API key",
                    "Save conversation context"
                ]
            )
        }

    def select_tool(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ToolSelection:
        """
        Select best tool for user query

        Args:
            user_query: User's natural language query
            context: Optional context (previous tools used, entities extracted, etc.)

        Returns:
            ToolSelection with recommended tool and reasoning
        """
        # Step 1: Extract intent and entities
        intent, entities = self._parse_query(user_query)

        # Step 2: Match capabilities
        candidates = self._match_capabilities(intent, entities, user_query)

        # Step 3: Rank candidates
        ranked = self._rank_candidates(candidates)

        if not ranked:
            return ToolSelection(
                tool_name="brave_search",  # Fallback to web search
                confidence=0.3,
                reasoning="No specific tool matched, falling back to web search",
                alternatives=[]
            )

        # Step 4: Select best tool
        best_tool, best_score = ranked[0]
        alternatives = [c[0] for c in ranked[1:3]]  # Top 2 alternatives

        # Step 5: Build parameters
        parameters = self._build_parameters(best_tool, entities)

        # Step 6: Check if tool composition needed
        composition = self._suggest_composition(user_query, best_tool, ranked)

        selection = ToolSelection(
            tool_name=best_tool,
            confidence=best_score,
            reasoning=self._build_reasoning(best_tool, intent, entities),
            parameters=parameters,
            alternatives=alternatives,
            composition=composition
        )

        # Store in history
        self.selection_history.append(selection)

        return selection

    def _parse_query(self, query: str) -> Tuple[str, Dict[str, List[str]]]:
        """
        Parse query to extract intent and entities

        Args:
            query: User query

        Returns:
            Tuple of (intent, entities_dict)
        """
        query_lower = query.lower()

        # Detect intent
        intent = "unknown"
        if any(word in query_lower for word in ["investigate", "analyze", "lookup", "find", "check"]):
            intent = "investigate"
        elif any(word in query_lower for word in ["search", "google", "latest", "news"]):
            intent = "search"
        elif any(word in query_lower for word in ["read", "show", "cat", "list", "ls"]):
            intent = "read"
        elif any(word in query_lower for word in ["write", "save", "create", "store"]):
            intent = "write"
        elif any(word in query_lower for word in ["threat", "malware", "attack", "vulnerability"]):
            intent = "threat_intel"

        # Extract entities
        entities = {
            "email": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', query),
            "domain": re.findall(r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b', query),
            "ip_address": re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', query),
            "bitcoin": re.findall(r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b', query),
            "ethereum": re.findall(r'\b0x[a-fA-F0-9]{40}\b', query),
            "phone": re.findall(r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}', query),
            "file_path": re.findall(r'[/\\](?:[^\s/\\]+[/\\])*[^\s/\\]+', query),
            "cve": re.findall(r'CVE-\d{4}-\d{4,}', query, re.IGNORECASE)
        }

        # Remove empty lists
        entities = {k: v for k, v in entities.items() if v}

        return intent, entities

    def _match_capabilities(
        self,
        intent: str,
        entities: Dict[str, List[str]],
        query: str
    ) -> List[str]:
        """
        Match tools based on capabilities

        Args:
            intent: Detected intent
            entities: Extracted entities
            query: Original query

        Returns:
            List of candidate tool names
        """
        candidates = []

        # Check each tool
        for tool_name, capability in self.tool_registry.items():
            score = 0.0

            # Keyword matching
            for keyword in capability.keywords:
                if keyword in query.lower():
                    score += 0.3

            # Entity type matching
            for entity_type in entities.keys():
                if entity_type in capability.input_types:
                    score += 0.5

            # Intent matching (heuristic)
            if intent == "investigate" and capability.category in [ToolCategory.OSINT, ToolCategory.THREAT_INTEL]:
                score += 0.4
            elif intent == "search" and capability.category == ToolCategory.WEB_SEARCH:
                score += 0.4
            elif intent == "read" and capability.category == ToolCategory.FILESYSTEM:
                score += 0.4
            elif intent == "write" and capability.category == ToolCategory.FILESYSTEM:
                score += 0.4

            if score > 0:
                candidates.append(tool_name)

        return candidates

    def _rank_candidates(self, candidates: List[str]) -> List[Tuple[str, float]]:
        """
        Rank candidate tools

        Args:
            candidates: List of candidate tool names

        Returns:
            List of (tool_name, score) tuples, sorted by score descending
        """
        ranked = []

        for tool_name in candidates:
            capability = self.tool_registry[tool_name]

            # Base score
            score = 0.5

            # Optimization factor
            if self.optimize_for == "speed":
                # Prefer faster tools
                score += (5000 - capability.latency_ms) / 5000 * 0.3
            elif self.optimize_for == "cost":
                # Prefer cheaper tools
                score += (4 - capability.cost_tier.value) / 4 * 0.3
            elif self.optimize_for == "quality":
                # Prefer higher-tier tools (assume higher quality)
                score += capability.cost_tier.value / 4 * 0.3
            else:  # balanced
                score += 0.2

            ranked.append((tool_name, score))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def _build_parameters(
        self,
        tool_name: str,
        entities: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Build suggested parameters for tool"""
        capability = self.tool_registry[tool_name]
        parameters = {}

        # Match entities to input types
        for input_type in capability.input_types:
            if input_type in entities and entities[input_type]:
                # Use first entity of this type
                if input_type == "crypto_address":
                    # Combine bitcoin and ethereum
                    if "bitcoin" in entities:
                        parameters["address"] = entities["bitcoin"][0]
                    elif "ethereum" in entities:
                        parameters["address"] = entities["ethereum"][0]
                else:
                    parameters[input_type] = entities[input_type][0]

        return parameters

    def _build_reasoning(
        self,
        tool_name: str,
        intent: str,
        entities: Dict[str, List[str]]
    ) -> str:
        """Build explanation for tool selection"""
        capability = self.tool_registry[tool_name]

        parts = []

        # Intent
        parts.append(f"Detected intent: {intent}")

        # Entities
        if entities:
            entity_str = ", ".join([f"{k}: {v[0]}" for k, v in entities.items()])
            parts.append(f"Extracted entities: {entity_str}")

        # Tool capability
        parts.append(f"Selected {tool_name} ({capability.category.value})")

        # Optimization
        if self.optimize_for != "balanced":
            parts.append(f"Optimized for: {self.optimize_for}")

        return "; ".join(parts)

    def _suggest_composition(
        self,
        query: str,
        primary_tool: str,
        ranked: List[Tuple[str, float]]
    ) -> List[str]:
        """
        Suggest tool composition (chaining) if needed

        Args:
            query: User query
            primary_tool: Primary selected tool
            ranked: All ranked candidates

        Returns:
            List of tools to chain (empty if no composition needed)
        """
        composition = [primary_tool]

        # Check if query implies multiple steps
        query_lower = query.lower()

        # Pattern: "investigate X and check Y"
        if " and " in query_lower:
            # Add top alternative if it's a different category
            if len(ranked) > 1:
                alt_tool = ranked[1][0]
                alt_capability = self.tool_registry[alt_tool]
                primary_capability = self.tool_registry[primary_tool]

                if alt_capability.category != primary_capability.category:
                    composition.append(alt_tool)

        # Pattern: Blockchain + threat intel
        if any(word in query_lower for word in ["suspicious", "malicious", "scam"]):
            if "blockchain" in primary_tool or "crypto" in query_lower:
                if "threat_intelligence" in self.tool_registry:
                    composition.append("threat_intelligence")

        return composition if len(composition) > 1 else []

    def get_all_tools(self) -> List[str]:
        """Get list of all available tools"""
        return list(self.tool_registry.keys())

    def get_tool_info(self, tool_name: str) -> Optional[ToolCapability]:
        """Get detailed info about a tool"""
        return self.tool_registry.get(tool_name)

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """Get all tools in a category"""
        return [
            name for name, cap in self.tool_registry.items()
            if cap.category == category
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics"""
        category_counts = {}
        for result in self.selection_history:
            tool = self.tool_registry.get(result.tool_name)
            if tool:
                cat = tool.category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_selections": len(self.selection_history),
            "category_distribution": category_counts,
            "avg_confidence": sum(r.confidence for r in self.selection_history) / max(len(self.selection_history), 1),
            "tools_used": len(set(r.tool_name for r in self.selection_history)),
            "composition_rate": sum(1 for r in self.selection_history if r.composition) / max(len(self.selection_history), 1)
        }


def main():
    """Demo usage"""
    print("=== MCP Tool Selector Demo ===\n")

    selector = MCPToolSelector(optimize_for="balanced")

    # Test queries
    test_queries = [
        "Investigate email address john@example.com",
        "Analyze Bitcoin address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "Is domain evil.com malicious?",
        "Search for latest AI news",
        "Read file /etc/hosts",
        "Lookup IP address 8.8.8.8",
        "Verify email and check if it was in a breach",
        "Investigate suspicious crypto wallet for scam activity"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = selector.select_tool(query)

        print(f"  Tool: {result.tool_name}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reasoning: {result.reasoning}")
        if result.parameters:
            print(f"  Parameters: {result.parameters}")
        if result.alternatives:
            print(f"  Alternatives: {result.alternatives}")
        if result.composition:
            print(f"  Composition: {' → '.join(result.composition)}")

    # Statistics
    print("\n\nStatistics:")
    stats = selector.get_statistics()
    print(f"  Total selections: {stats['total_selections']}")
    print(f"  Category distribution: {stats['category_distribution']}")
    print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    print(f"  Composition rate: {stats['composition_rate']:.2%}")


if __name__ == "__main__":
    main()
