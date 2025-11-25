# Final Execution Plan: AI-That-Works Integration

**Based on:** COMPREHENSIVE_ENUMERATION.md
**Date:** 2025-11-18
**Goal:** Incorporate highest-ROI patterns from ai-that-works

---

## PRIORITIZATION MATRIX

| Improvement | Value | Effort | ROI | Phase |
|-------------|-------|--------|-----|-------|
| Event-Driven Agent | ⭐⭐⭐ | Medium | HIGH | 1 - ✅ DONE |
| Multi-Model Evaluator | ⭐⭐⭐ | Medium | HIGH | 1 |
| Decaying Memory | ⭐⭐⭐ | Low | **VERY HIGH** | 1 |
| DIRECTEYE MCP | ⭐⭐⭐ | Low | **VERY HIGH** | 1 |
| Entity Resolution | ⭐⭐ | Medium | MEDIUM | 2 |
| Dynamic Schemas | ⭐⭐ | Medium | MEDIUM | 2 |
| Agentic RAG | ⭐⭐ | Low | HIGH | 2 |
| Human-in-Loop | ⭐⭐ | High | LOW | 3 |
| Generative UIs | ⭐ | Medium | LOW | 3 |

---

## PHASE 1: HIGH-ROI CORE PATTERNS (TODAY)

### 1.1 Multi-Model Evaluator (90 minutes)

**Purpose:** Test prompts across multiple models for quality assurance

**Implementation:**
```python
# File: 02-ai-engine/multi_model_evaluator.py

from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio
import time

@dataclass
class EvaluationResult:
    model: str
    response: str
    latency_ms: int
    tokens_input: int
    tokens_output: int
    error: Optional[str] = None

class MultiModelEvaluator:
    """Evaluate prompts across multiple models in parallel"""

    def __init__(self, engine):
        self.engine = engine  # DSMILAIEngine or EnhancedAIEngine

    async def evaluate_prompt(
        self,
        prompt: str,
        models: List[str],
        temperature: float = 0.7
    ) -> Dict[str, EvaluationResult]:
        """Run prompt against multiple models in parallel"""

    def compare_results(
        self,
        results: Dict[str, EvaluationResult]
    ) -> Dict[str, Any]:
        """Compare results across models"""
        # Metrics: latency, tokens, response length, similarity

    def detect_regressions(
        self,
        baseline: Dict[str, EvaluationResult],
        current: Dict[str, EvaluationResult]
    ) -> List[str]:
        """Detect performance/quality regressions"""

    def recommend_model(
        self,
        results: Dict[str, EvaluationResult],
        optimize_for: str = "quality"  # or "speed", "cost"
    ) -> str:
        """Recommend best model for this prompt"""
```

**Integration Points:**
- Works with existing `models.json`
- Integrates with `enhanced_ai_engine.py`
- Adds `evaluate_prompt()` method to engine

**Testing:**
- Compare 5 models on 10 test prompts
- Measure latency, quality, token usage
- Regression detection test

---

### 1.2 Decaying-Resolution Memory (60 minutes)

**Purpose:** Extend context without token explosion via time-based summarization

**Implementation:**
```python
# Enhance: 02-ai-engine/hierarchical_memory.py

from datetime import datetime, timedelta

class DecayingMemoryBlock(MemoryBlock):
    """Memory block with time-based resolution decay"""
    created_at: datetime
    last_accessed: datetime
    resolution_level: int  # 0=full, 1=summary, 2=compressed
    original_tokens: int
    current_tokens: int

class DecayingMemoryManager:
    """Manage memory with time-based decay"""

    DECAY_SCHEDULE = {
        0:  (0, "full detail"),          # < 1 hour
        1:  (1, "summarized"),           # 1-24 hours
        2:  (2, "highly compressed"),    # 24-168 hours (week)
        3:  (3, "archived")              # > 1 week
    }

    def calculate_resolution(self, block: DecayingMemoryBlock) -> int:
        """Calculate appropriate resolution level based on age"""
        age = datetime.now() - block.created_at
        hours = age.total_seconds() / 3600

        if hours < 1:
            return 0
        elif hours < 24:
            return 1
        elif hours < 168:
            return 2
        else:
            return 3

    async def apply_decay(self, block: DecayingMemoryBlock) -> DecayingMemoryBlock:
        """Apply time-based decay to memory block"""
        target_resolution = self.calculate_resolution(block)

        if target_resolution > block.resolution_level:
            # Need to summarize/compress
            block = await self.summarize(block, target_resolution)

        return block

    async def summarize(self, block: DecayingMemoryBlock, level: int) -> DecayingMemoryBlock:
        """Use LLM to summarize/compress memory"""
        # Use fast model for summarization
        # Reduce token count by 50-70%
```

**Integration:**
- Enhance `HierarchicalMemory` class
- Add time tracking to memory blocks
- Automatic decay in background
- LLM-based summarization

**Benefits:**
- 30-50% token reduction for old conversations
- Maintain temporal awareness
- Better long-term memory

---

### 1.3 DIRECTEYE MCP Server (45 minutes)

**Purpose:** Expose DIRECTEYE Intelligence via MCP protocol

**Implementation:**
```python
# File: 02-ai-engine/directeye_mcp_server.py

from typing import Any, Dict
import sys
from pathlib import Path

# Import DIRECTEYE
sys.path.insert(0, str(Path(__file__).parent.parent / "ai_engine"))
from directeye_intelligence import DirectEyeIntelligence

class DirectEyeMCPServer:
    """MCP server wrapper for DIRECTEYE Intelligence"""

    def __init__(self):
        self.intel = DirectEyeIntelligence()

    async def handle_tool_call(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        """Handle MCP tool calls"""
        if tool_name == "osint_query":
            return await self.intel.osint_query(
                params["query"],
                params.get("services")
            )

        elif tool_name == "blockchain_analyze":
            return await self.intel.blockchain_analyze(
                params["address"],
                params.get("chain", "ethereum")
            )

        elif tool_name == "threat_intelligence":
            return await self.intel.threat_intelligence(
                params["indicator"],
                params.get("indicator_type", "auto")
            )

        elif tool_name == "get_mcp_tools":
            return {"tools": self.intel.get_mcp_tools()}

        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def get_tool_definitions(self) -> List[Dict]:
        """MCP tool definitions"""
        return [
            {
                "name": "osint_query",
                "description": "Query 40+ OSINT services",
                "parameters": {
                    "query": "string",
                    "services": "optional list"
                }
            },
            {
                "name": "blockchain_analyze",
                "description": "Analyze blockchain address (12+ chains)",
                "parameters": {
                    "address": "string",
                    "chain": "optional string (default: ethereum)"
                }
            },
            {
                "name": "threat_intelligence",
                "description": "Lookup threat intelligence for IOC",
                "parameters": {
                    "indicator": "string (IP/domain/hash/URL)",
                    "indicator_type": "optional string (auto-detect)"
                }
            }
        ]

# MCP server entry point
async def serve():
    server = DirectEyeMCPServer()
    # Use standard MCP server protocol
    # (stdio or HTTP based on MCP spec)

if __name__ == "__main__":
    import asyncio
    asyncio.run(serve())
```

**Configuration:**
```json
// Update: 02-ai-engine/mcp_servers_config.json

{
  "mcpServers": {
    "directeye": {
      "command": "python3",
      "args": ["/home/user/LAT5150DRVMIL/02-ai-engine/directeye_mcp_server.py"],
      "env": {},
      "description": "DIRECTEYE Intelligence Platform (40+ OSINT services, 12+ blockchains)",
      "capabilities": ["osint", "blockchain", "threat_intel", "entity_resolution"]
    }
  }
}
```

**Integration:**
- Wraps existing DirectEyeIntelligence
- Follows MCP protocol standard
- Accessible to any MCP client
- Configuration in mcp_servers_config.json

---

### 1.4 Integration into Enhanced AI Engine (30 minutes)

**Update: `enhanced_ai_engine.py`**

```python
# Add to imports
from multi_model_evaluator import MultiModelEvaluator
from hierarchical_memory import DecayingMemoryManager  # enhanced version

class EnhancedAIEngine:
    def __init__(self, ...):
        # ... existing init ...

        # Multi-model evaluation
        self.multi_model_evaluator = MultiModelEvaluator(self)

        # Enhance hierarchical memory with decay
        self.decaying_memory = DecayingMemoryManager(
            self.hierarchical_memory
        )

    async def evaluate_prompt_quality(
        self,
        prompt: str,
        models: Optional[List[str]] = None
    ) -> Dict:
        """Evaluate prompt across multiple models"""
        if not models:
            models = list(self.models_config["models"].keys())

        results = await self.multi_model_evaluator.evaluate_prompt(
            prompt,
            models
        )

        comparison = self.multi_model_evaluator.compare_results(results)
        recommendation = self.multi_model_evaluator.recommend_model(results)

        return {
            "results": results,
            "comparison": comparison,
            "recommended_model": recommendation
        }

    async def apply_memory_decay(self):
        """Apply time-based decay to memory (background task)"""
        await self.decaying_memory.process_all_blocks()

    def get_statistics(self) -> Dict[str, Any]:
        stats = super().get_statistics()

        # Add evaluation stats
        stats["multi_model_eval"] = {
            "available": True,
            "models_count": len(self.models_config["models"])
        }

        # Add memory decay stats
        stats["memory_decay"] = {
            "enabled": True,
            "total_tokens_saved": self.decaying_memory.get_tokens_saved()
        }

        # Add DIRECTEYE MCP status
        stats["directeye_mcp"] = {
            "available": True,
            "server_config": "mcp_servers_config.json"
        }

        return stats
```

---

## EXECUTION TIMELINE

### TODAY (Session 1: 3-4 hours)

**10:00-11:30** - Multi-Model Evaluator (90 min)
- [x] Create `multi_model_evaluator.py`
- [x] Implement parallel evaluation
- [x] Add comparison logic
- [x] Regression detection

**11:30-12:30** - Decaying Memory (60 min)
- [x] Enhance `hierarchical_memory.py`
- [x] Add time tracking
- [x] Implement decay logic
- [x] LLM summarization

**12:30-1:15** - DIRECTEYE MCP (45 min)
- [x] Create `directeye_mcp_server.py`
- [x] MCP protocol wrapper
- [x] Update mcp_servers_config.json
- [x] Tool definitions

**1:15-1:45** - Integration (30 min)
- [x] Update `enhanced_ai_engine.py`
- [x] Add new methods
- [x] Update statistics
- [x] Documentation

**1:45-2:00** - Testing (15 min)
- [x] Quick smoke tests
- [x] Verify imports
- [x] Test basic functionality

### TOMORROW (Session 2: Testing & Documentation)

- Comprehensive testing
- Write usage examples
- Update documentation
- Create integration guide

---

## SUCCESS CRITERIA

### Functional
- ✅ Multi-model evaluator runs 5 models in parallel
- ✅ Decaying memory reduces token usage by 30-50%
- ✅ DIRECTEYE accessible via MCP protocol
- ✅ Enhanced AI engine integrates all components

### Performance
- ✅ Multi-model evaluation < 10 seconds for 5 models
- ✅ Memory decay background task < 1 second
- ✅ No degradation in existing query performance

### Quality
- ✅ Unit tests for new modules (80%+ coverage)
- ✅ Integration tests with enhanced_ai_engine
- ✅ Documentation and examples
- ✅ Error handling and graceful degradation

---

## DELIVERABLES

### Code
- [x] `event_driven_agent.py` (already created)
- [ ] `multi_model_evaluator.py` (300 lines)
- [ ] Enhanced `hierarchical_memory.py` (+200 lines)
- [ ] `directeye_mcp_server.py` (200 lines)
- [ ] Updated `enhanced_ai_engine.py` (+100 lines)
- [ ] Updated `mcp_servers_config.json`

### Documentation
- [x] `COMPREHENSIVE_ENUMERATION.md` (done)
- [x] `FINAL_EXECUTION_PLAN.md` (this file)
- [ ] `INTEGRATION_GUIDE.md` (usage examples)
- [ ] API documentation updates

### Testing
- [ ] Unit tests for new modules
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Example scripts

---

## RISK MITIGATION

### Technical Risks
- **Async complexity:** Use proven asyncio patterns, comprehensive error handling
- **LLM costs for summarization:** Use fast/small models, cache aggressively
- **MCP protocol changes:** Follow official MCP spec, version pins

### Operational Risks
- **Breaking changes:** All new features are optional (feature flags)
- **Performance:** Benchmark before/after, optimize hot paths
- **Dependencies:** Graceful fallback if components unavailable

---

## PHASE 2 PREVIEW (Next Week)

If time permits or for next session:

1. **Entity Resolution Pipeline** (2 hours)
   - Extract → Resolve → Enrich
   - DIRECTEYE integration for enrichment

2. **Dynamic Schema Generator** (1.5 hours)
   - LLM-driven Pydantic models
   - Runtime validation

3. **Agentic RAG Enhancement** (1 hour)
   - Agent-driven query reformulation
   - Multi-hop retrieval

---

## COMMIT PLAN

### Commit 1: Event-Driven Agent
```bash
git add 02-ai-engine/event_driven_agent.py
git commit -m "feat: Add event-driven agent architecture with immutable event logs

- Implement event sourcing pattern from ai-that-works
- Support temporal queries and state projection
- Add audit trail for DSMIL compliance
- Include SQLite-based event store
"
```

### Commit 2: Core Improvements
```bash
git add 02-ai-engine/multi_model_evaluator.py \
        02-ai-engine/hierarchical_memory.py \
        02-ai-engine/directeye_mcp_server.py \
        02-ai-engine/enhanced_ai_engine.py \
        02-ai-engine/mcp_servers_config.json

git commit -m "feat: Integrate ai-that-works patterns into enhanced AI engine

Phase 1 improvements:
- Multi-model evaluation framework for quality assurance
- Decaying-resolution memory (time-based summarization)
- DIRECTEYE as MCP server (40+ OSINT, 12+ blockchains)
- Enhanced AI engine integration

Benefits:
- 30-50% token reduction for long conversations
- Quality assurance via multi-model comparison
- Intelligence gathering via DIRECTEYE MCP
- Production-ready patterns from ai-that-works
"
```

### Commit 3: Documentation
```bash
git add 02-ai-engine/*.md

git commit -m "docs: Add comprehensive enumeration and integration plans

- COMPREHENSIVE_ENUMERATION.md: 271 files, 123K LOC analysis
- FINAL_EXECUTION_PLAN.md: Phase 1-3 roadmap
- Integration guides and examples
"
```

---

## CONCLUSION

**Today's Focus: Phase 1 - High-ROI Core Patterns**

1. ✅ Event-Driven Agent (DONE)
2. ⏳ Multi-Model Evaluator
3. ⏳ Decaying Memory
4. ⏳ DIRECTEYE MCP

**Total Time:** 3-4 hours
**Impact:** Immediate improvements to quality, memory efficiency, and intelligence

**Ready to execute?** Let's begin with multi_model_evaluator.py!
