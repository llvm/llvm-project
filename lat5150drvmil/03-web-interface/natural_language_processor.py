#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Natural Language Command Processor
LOCAL-FIRST: Uses local Ollama models for understanding and routing

Maps natural language commands to system capabilities without external API calls
"""

import json
import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

# Local imports
from capability_registry import get_registry, Capability
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../01-source'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] NLP: %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from agentsystems_integration.model_providers import OllamaProvider
except ImportError:
    logger.warning("Model providers not available - using rule-based fallback")
    OllamaProvider = None


@dataclass
class ParsedCommand:
    """Parsed natural language command"""
    original_query: str
    intent: str
    matched_capability: Optional[Capability]
    extracted_parameters: Dict[str, Any]
    confidence: float
    alternative_capabilities: List[Capability]

    def to_dict(self) -> Dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "matched_capability": self.matched_capability.to_dict() if self.matched_capability else None,
            "extracted_parameters": self.extracted_parameters,
            "confidence": self.confidence,
            "alternative_capabilities": [cap.to_dict() for cap in self.alternative_capabilities]
        }


class NaturalLanguageProcessor:
    """
    LOCAL-FIRST Natural Language Processor

    Uses local Ollama models for understanding, with rule-based fallback
    """

    def __init__(
        self,
        use_local_model: bool = True,
        ollama_endpoint: str = "http://localhost:11434",
        model: str = "llama3.2:latest"
    ):
        self.registry = get_registry()
        self.use_local_model = use_local_model
        self.ollama_endpoint = ollama_endpoint
        self.model = model

        # Initialize local model provider
        self.ollama_provider = None
        if use_local_model and OllamaProvider:
            try:
                self.ollama_provider = OllamaProvider(endpoint=ollama_endpoint)
                logger.info(f"✅ LOCAL MODEL: Using Ollama {model} at {ollama_endpoint}")
            except Exception as e:
                logger.warning(f"⚠️  Ollama not available, using rule-based fallback: {e}")

        # Build rule-based patterns
        self._build_patterns()

    def _build_patterns(self):
        """Build rule-based patterns for command matching"""
        self.patterns = []

        for cap_id, capability in self.registry.capabilities.items():
            for trigger in capability.natural_language_triggers:
                # Create regex pattern
                pattern = re.escape(trigger).replace(r'\ ', r'\s+')
                self.patterns.append({
                    "pattern": re.compile(pattern, re.IGNORECASE),
                    "capability_id": cap_id,
                    "trigger": trigger
                })

    async def parse_command(self, query: str) -> ParsedCommand:
        """
        Parse natural language command

        Args:
            query: Natural language command

        Returns:
            ParsedCommand with matched capability and parameters
        """
        logger.info(f"Parsing command: '{query}'")

        # Method 1: Try local model (if available)
        if self.ollama_provider:
            try:
                result = await self._parse_with_local_model(query)
                if result and result.confidence > 0.6:
                    logger.info(f"✅ LOCAL MODEL matched: {result.matched_capability.name if result.matched_capability else 'None'} (confidence: {result.confidence:.2f})")
                    return result
            except Exception as e:
                logger.warning(f"Local model parsing failed: {e}")

        # Method 2: Rule-based fallback
        result = await self._parse_with_rules(query)
        logger.info(f"✅ RULE-BASED matched: {result.matched_capability.name if result.matched_capability else 'None'} (confidence: {result.confidence:.2f})")
        return result

    async def _parse_with_local_model(self, query: str) -> Optional[ParsedCommand]:
        """Parse using local Ollama model"""

        # Create capability context for the model
        capabilities_context = self._build_capabilities_context()

        prompt = f"""You are a command parser for a military-grade AI system. Parse the user's natural language command and map it to one of the available capabilities.

Available Capabilities:
{capabilities_context}

User Command: "{query}"

Respond with ONLY a JSON object (no markdown, no explanation):
{{
  "matched_capability_id": "capability_id",
  "confidence": 0.95,
  "extracted_parameters": {{"param1": "value1"}},
  "reasoning": "brief explanation"
}}

If no good match, set matched_capability_id to null."""

        try:
            response = await self.ollama_provider.complete(
                prompt=prompt,
                model=self.model,
                temperature=0.1  # Low temperature for consistency
            )

            # Parse JSON response
            response_text = response.text.strip()

            # Try to extract JSON (model might add markdown)
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)

            parsed = json.loads(response_text)

            matched_cap_id = parsed.get("matched_capability_id")
            if not matched_cap_id:
                return None

            matched_capability = self.registry.get(matched_cap_id)
            if not matched_capability:
                return None

            return ParsedCommand(
                original_query=query,
                intent=parsed.get("reasoning", ""),
                matched_capability=matched_capability,
                extracted_parameters=parsed.get("extracted_parameters", {}),
                confidence=float(parsed.get("confidence", 0.0)),
                alternative_capabilities=[]
            )

        except Exception as e:
            logger.error(f"Error parsing with local model: {e}")
            return None

    async def _parse_with_rules(self, query: str) -> ParsedCommand:
        """Parse using rule-based pattern matching"""

        query_lower = query.lower()
        matches = []

        # Try pattern matching
        for pattern_info in self.patterns:
            if pattern_info["pattern"].search(query_lower):
                capability = self.registry.get(pattern_info["capability_id"])
                if capability:
                    matches.append({
                        "capability": capability,
                        "score": 1.0,
                        "trigger": pattern_info["trigger"]
                    })

        # If no pattern matches, try semantic search
        if not matches:
            semantic_matches = self.registry.search_by_natural_language(query)
            for cap in semantic_matches[:5]:
                matches.append({
                    "capability": cap,
                    "score": 0.7,
                    "trigger": "semantic_match"
                })

        if not matches:
            # No match found
            return ParsedCommand(
                original_query=query,
                intent="unknown",
                matched_capability=None,
                extracted_parameters={},
                confidence=0.0,
                alternative_capabilities=[]
            )

        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)

        # Best match
        best_match = matches[0]
        matched_capability = best_match["capability"]

        # Extract parameters (simple extraction)
        extracted_params = self._extract_parameters(query, matched_capability)

        # Alternative capabilities
        alternatives = [m["capability"] for m in matches[1:3]]

        return ParsedCommand(
            original_query=query,
            intent=f"Matched via {best_match['trigger']}",
            matched_capability=matched_capability,
            extracted_parameters=extracted_params,
            confidence=best_match["score"],
            alternative_capabilities=alternatives
        )

    def _extract_parameters(
        self,
        query: str,
        capability: Capability
    ) -> Dict[str, Any]:
        """Extract parameters from query for a capability"""
        params = {}

        # Simple parameter extraction patterns
        patterns = {
            # File paths
            r'file\s+([^\s]+\.py)': 'file_path',
            r'path\s+([^\s]+)': 'file_path',

            # Device IDs
            r'0x([0-9a-fA-F]{4})': 'device_id',
            r'device\s+([0-9]+)': 'device_id',

            # Symbol names
            r'function\s+(\w+)': 'name',
            r'class\s+(\w+)': 'name',
            r'variable\s+(\w+)': 'name',
            r'symbol\s+(\w+)': 'name',

            # Agent names
            r'agent\s+(\S+)': 'agent_name',

            # Thread IDs
            r'thread-[\w-]+': 'thread_id',

            # Models
            r'claude|gpt|llama|mixtral': 'model',

            # Modes
            r'level\s*[a-c]': 'mode',
            r'comfort|night|nvg|contrast': 'mode',
        }

        for pattern, param_name in patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                if match.groups():
                    params[param_name] = match.group(1)
                else:
                    params[param_name] = match.group(0)

        return params

    def _build_capabilities_context(self) -> str:
        """Build context string of capabilities for model"""
        lines = []

        for cap_id, capability in self.registry.capabilities.items():
            lines.append(f"- {cap_id}: {capability.name}")
            lines.append(f"  Description: {capability.description}")
            lines.append(f"  Triggers: {', '.join(capability.natural_language_triggers[:3])}")
            lines.append(f"  Parameters: {', '.join(capability.parameters.keys())}")
            lines.append("")

        return '\n'.join(lines[:500])  # Limit context size

    def get_help_text(self, query: Optional[str] = None) -> str:
        """
        Get help text for available commands

        Args:
            query: Optional query to filter help

        Returns:
            Help text
        """
        if query:
            capabilities = self.registry.search_by_natural_language(query)[:5]
        else:
            capabilities = self.registry.list_all()

        lines = ["🤖 LAT5150 DRVMIL Tactical AI - Available Commands\n"]

        # Group by category
        from collections import defaultdict
        by_category = defaultdict(list)

        for cap in capabilities:
            by_category[cap.category.value].append(cap)

        for category, caps in sorted(by_category.items()):
            lines.append(f"\n📂 {category.upper().replace('_', ' ')}")
            lines.append("=" * 60)

            for cap in caps:
                lines.append(f"\n• {cap.name}")
                lines.append(f"  {cap.description[:80]}...")
                lines.append(f"  Examples:")
                for example in cap.examples[:2]:
                    lines.append(f"    - {example}")

        return '\n'.join(lines)


# Example usage
async def main():
    """Test natural language processor"""

    print("\n" + "="*70)
    print("LAT5150 DRVMIL - Natural Language Processor")
    print("LOCAL-FIRST: Using Ollama models")
    print("="*70 + "\n")

    # Initialize processor
    nlp = NaturalLanguageProcessor(
        use_local_model=True,
        ollama_endpoint="http://localhost:11434",
        model="llama3.2:latest"
    )

    # Test queries
    test_queries = [
        "Find the NSADeviceReconnaissance class",
        "Scan for DSMIL devices",
        "Run the security analyzer agent",
        "What's the system health?",
        "Switch to Level A TEMPEST mode",
        "Find all references to process_data",
        "Use llama3.2 to analyze this code",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('-'*70)

        result = await nlp.parse_command(query)

        if result.matched_capability:
            print(f"✅ Matched: {result.matched_capability.name}")
            print(f"   Confidence: {result.confidence:.2f}")
            print(f"   Category: {result.matched_capability.category.value}")

            if result.extracted_parameters:
                print(f"   Parameters:")
                for key, value in result.extracted_parameters.items():
                    print(f"     - {key}: {value}")

            if result.alternative_capabilities:
                print(f"   Alternatives:")
                for alt in result.alternative_capabilities:
                    print(f"     - {alt.name}")
        else:
            print("❌ No match found")
            print(f"   Try: /help {query}")

    # Show help
    print("\n" + "="*70)
    print("HELP: Code Understanding Commands")
    print("="*70)
    help_text = nlp.get_help_text("find code")
    print(help_text[:500] + "...")


if __name__ == "__main__":
    asyncio.run(main())
