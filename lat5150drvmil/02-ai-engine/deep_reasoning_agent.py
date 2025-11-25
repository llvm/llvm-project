#!/usr/bin/env python3
"""
Deep Reasoning Agent - Inspired by DeepAgent (arXiv 2510.21618)

Implements:
- Autonomous thinking within a single reasoning process
- Dynamic tool discovery with dense indexing
- Memory folding for long-horizon tasks
- ToolPO-style learning from tool usage

Integration with our Enhanced AI Engine:
- Uses our 11 MCP servers
- Leverages our hierarchical memory
- Works with our RAG system
- Integrates with self-improvement
"""

import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class Tool:
    """Tool/MCP server definition"""
    name: str
    description: str
    category: str
    parameters: List[str]
    example_usage: str
    success_rate: float = 0.0  # Learned from usage
    avg_latency_ms: int = 0     # Learned from usage
    use_count: int = 0           # Usage tracking
    embedding: Optional[np.ndarray] = None  # Dense vector


@dataclass
class MemoryFold:
    """Compressed memory representation"""
    fold_id: str
    fold_type: str  # episodic, working, tool
    original_length: int
    compressed_length: int
    content: str
    timestamp: datetime
    importance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningStep:
    """Single step in reasoning process"""
    step_num: int
    step_type: str  # thinking, tool_discovery, tool_execution, reflection
    content: str
    tools_considered: List[str]
    tool_selected: Optional[str]
    tool_result: Optional[str]
    confidence: float
    latency_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for a task"""
    task_id: str
    task_prompt: str
    steps: List[ReasoningStep]
    final_answer: str
    total_latency_ms: int
    tools_used: List[str]
    memory_folds: List[MemoryFold]
    success: bool
    quality_score: float


class DeepReasoningAgent:
    """
    Deep Reasoning Agent with autonomous thinking and tool discovery

    Key Features (from DeepAgent):
    1. Unified reasoning process (not workflow-based)
    2. Autonomous memory folding (compress long interactions)
    3. Dense tool discovery (not hardcoded)
    4. ToolPO-style learning (improve from usage)

    Integration with our system:
    - 11 MCP servers as tool ecosystem
    - Hierarchical memory for memory folding
    - RAG system for context retrieval
    - Self-improvement for learning
    """

    def __init__(
        self,
        mcp_config_path: str = "/home/user/LAT5150DRVMIL/02-ai-engine/mcp_servers_config.json",
        storage_dir: str = "/home/user/LAT5150DRVMIL/02-ai-engine/deep_reasoning_data"
    ):
        """
        Initialize Deep Reasoning Agent

        Args:
            mcp_config_path: Path to MCP servers config
            storage_dir: Directory for learning data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Load tool ecosystem
        self.tools: Dict[str, Tool] = self._load_tools(mcp_config_path)

        # Initialize embedder for dense tool discovery
        self.embedder = None
        if SentenceTransformer:
            try:
                print("🔍 Loading embedding model for tool discovery...")
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                self._embed_tools()
            except Exception as e:
                print(f"⚠️  Embedding model failed: {e}")

        # Memory folding state
        self.episodic_memory: List[str] = []  # Full interaction history
        self.working_memory: List[str] = []   # Current task context
        self.tool_memory: Dict[str, Any] = {}  # Learned tool patterns
        self.memory_folds: List[MemoryFold] = []

        # Reasoning traces for learning
        self.reasoning_traces: List[ReasoningTrace] = []

        # Load learned patterns
        self._load_tool_memory()

        print(f"✅ Deep Reasoning Agent initialized with {len(self.tools)} tools")

    def _load_tools(self, config_path: str) -> Dict[str, Tool]:
        """Load tools from MCP servers config"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        tools = {}

        for name, mcp_config in config.get("mcpServers", {}).items():
            description = mcp_config.get("description", "No description")

            # Categorize tools
            category = self._categorize_tool(name, description)

            # Extract example usage
            example = self._generate_example_usage(name, description)

            tools[name] = Tool(
                name=name,
                description=description,
                category=category,
                parameters=[],  # Would need to introspect MCP server
                example_usage=example,
                success_rate=0.5,  # Start neutral
                avg_latency_ms=1000,  # Assume 1s
                use_count=0
            )

        return tools

    def _categorize_tool(self, name: str, description: str) -> str:
        """Categorize tool by purpose"""
        desc_lower = description.lower()
        name_lower = name.lower()

        if any(w in desc_lower or w in name_lower for w in ["security", "scan", "metasploit", "nmap"]):
            return "security"
        elif any(w in desc_lower or w in name_lower for w in ["search", "find", "grep", "code"]):
            return "code_search"
        elif any(w in desc_lower or w in name_lower for w in ["doc", "documentation", "reference"]):
            return "documentation"
        elif any(w in desc_lower or w in name_lower for w in ["osint", "investigation", "recon"]):
            return "osint"
        elif any(w in desc_lower or w in name_lower for w in ["file", "filesystem", "read", "write"]):
            return "filesystem"
        elif any(w in desc_lower or w in name_lower for w in ["memory", "remember", "recall"]):
            return "memory"
        elif any(w in desc_lower or w in name_lower for w in ["think", "reason", "sequential"]):
            return "reasoning"
        elif any(w in desc_lower or w in name_lower for w in ["ai", "llm", "model", "rag"]):
            return "ai_engine"
        else:
            return "general"

    def _generate_example_usage(self, name: str, description: str) -> str:
        """Generate example usage for tool"""
        examples = {
            "search-tools": "Search for 'authentication' in codebase",
            "docs-mcp-server": "Find documentation for React hooks",
            "metasploit": "Scan target 10.0.0.1 for vulnerabilities",
            "security-tools": "Run nmap port scan on 192.168.1.1",
            "maigret": "Search username 'johndoe' across social networks",
            "filesystem": "Read file /etc/passwd",
            "memory": "Remember user's favorite color is blue",
            "sequential-thinking": "Solve multi-step reasoning problem",
            "dsmil-ai": "Query AI model with RAG context"
        }

        return examples.get(name, f"Use {name} for: {description[:50]}...")

    def _embed_tools(self):
        """Create dense embeddings for all tools"""
        if not self.embedder:
            return

        for tool_name, tool in self.tools.items():
            # Create embedding text from name, description, category
            embed_text = f"{tool.name} {tool.category} {tool.description}"

            try:
                embedding = self.embedder.encode(embed_text, convert_to_numpy=True)
                tool.embedding = embedding
            except Exception as e:
                print(f"⚠️  Failed to embed tool {tool_name}: {e}")

    def _load_tool_memory(self):
        """Load learned tool usage patterns"""
        memory_file = self.storage_dir / "tool_memory.json"

        if memory_file.exists():
            with open(memory_file, 'r') as f:
                self.tool_memory = json.load(f)

            # Apply learned patterns to tools
            for tool_name, stats in self.tool_memory.items():
                if tool_name in self.tools:
                    self.tools[tool_name].success_rate = stats.get("success_rate", 0.5)
                    self.tools[tool_name].avg_latency_ms = stats.get("avg_latency_ms", 1000)
                    self.tools[tool_name].use_count = stats.get("use_count", 0)

    def _save_tool_memory(self):
        """Save learned tool usage patterns"""
        memory_file = self.storage_dir / "tool_memory.json"

        # Serialize tool stats
        tool_stats = {}
        for tool_name, tool in self.tools.items():
            tool_stats[tool_name] = {
                "success_rate": tool.success_rate,
                "avg_latency_ms": tool.avg_latency_ms,
                "use_count": tool.use_count
            }

        with open(memory_file, 'w') as f:
            json.dump(tool_stats, f, indent=2)

    def reason(
        self,
        task_prompt: str,
        max_steps: int = 20,
        thinking_budget: int = 5,
        fold_threshold: int = 10
    ) -> ReasoningTrace:
        """
        Main reasoning method - unified process (DeepAgent-style)

        Args:
            task_prompt: Task to solve
            max_steps: Maximum reasoning steps
            thinking_budget: Number of internal thinking steps before action
            fold_threshold: When to fold memory (number of steps)

        Returns:
            ReasoningTrace with complete reasoning process
        """
        task_id = hashlib.md5(task_prompt.encode()).hexdigest()[:16]
        start_time = time.time()

        print(f"\n{'='*70}")
        print(f"Deep Reasoning: {task_prompt[:60]}...")
        print(f"{'='*70}\n")

        steps: List[ReasoningStep] = []
        tools_used: List[str] = []
        self.working_memory = [f"Task: {task_prompt}"]

        current_context = task_prompt
        final_answer = ""

        for step_num in range(max_steps):
            print(f"📍 Step {step_num + 1}/{max_steps}")

            # Check if should fold memory
            if len(self.working_memory) > fold_threshold:
                self._fold_memory()

            # Phase 1: Extended thinking (DeepAgent's unified reasoning)
            thinking_step = self._thinking_phase(
                current_context,
                step_num,
                thinking_budget
            )
            steps.append(thinking_step)
            print(f"   💭 Thinking: {thinking_step.content[:80]}...")

            # Check if task is complete
            if self._is_task_complete(thinking_step.content):
                final_answer = thinking_step.content
                print(f"   ✅ Task complete!")
                break

            # Phase 2: Tool discovery (dense retrieval)
            if self._should_use_tool(thinking_step.content):
                discovery_step = self._tool_discovery_phase(
                    current_context,
                    thinking_step.content,
                    step_num
                )
                steps.append(discovery_step)

                if discovery_step.tool_selected:
                    print(f"   🔧 Selected tool: {discovery_step.tool_selected}")

                    # Phase 3: Tool execution
                    execution_step = self._tool_execution_phase(
                        discovery_step.tool_selected,
                        thinking_step.content,
                        step_num
                    )
                    steps.append(execution_step)

                    if execution_step.tool_result:
                        print(f"   ✅ Tool result: {execution_step.tool_result[:80]}...")
                        tools_used.append(discovery_step.tool_selected)

                        # Update context
                        current_context += f"\n\nTool {discovery_step.tool_selected} returned: {execution_step.tool_result}"
                        self.working_memory.append(f"Used {discovery_step.tool_selected}: {execution_step.tool_result[:100]}")

            # Phase 4: Reflection (periodic self-assessment)
            if (step_num + 1) % 5 == 0:
                reflection_step = self._reflection_phase(
                    current_context,
                    steps,
                    step_num
                )
                steps.append(reflection_step)
                print(f"   🤔 Reflection: {reflection_step.content[:80]}...")

            print()

        # If didn't complete, use last thinking as answer
        if not final_answer and steps:
            final_answer = steps[-1].content

        total_latency_ms = int((time.time() - start_time) * 1000)

        # Create reasoning trace
        trace = ReasoningTrace(
            task_id=task_id,
            task_prompt=task_prompt,
            steps=steps,
            final_answer=final_answer,
            total_latency_ms=total_latency_ms,
            tools_used=tools_used,
            memory_folds=self.memory_folds.copy(),
            success=bool(final_answer),
            quality_score=self._assess_quality(final_answer, task_prompt)
        )

        # Learn from this trace (ToolPO-style)
        self._learn_from_trace(trace)

        # Store for analysis
        self.reasoning_traces.append(trace)

        print(f"{'='*70}")
        print(f"✅ Reasoning complete: {total_latency_ms}ms, {len(steps)} steps")
        print(f"   Tools used: {', '.join(tools_used) if tools_used else 'None'}")
        print(f"   Memory folds: {len(self.memory_folds)}")
        print(f"{'='*70}\n")

        return trace

    def _thinking_phase(
        self,
        context: str,
        step_num: int,
        budget: int
    ) -> ReasoningStep:
        """Internal thinking without tools"""
        start_time = time.time()

        # Simulate deep thinking (in real impl, would call LLM)
        thinking_content = f"[Step {step_num}] Analyzing: {context[:100]}... "
        thinking_content += "Considering next action, evaluating options, planning approach."

        latency_ms = int((time.time() - start_time) * 1000)

        return ReasoningStep(
            step_num=step_num,
            step_type="thinking",
            content=thinking_content,
            tools_considered=[],
            tool_selected=None,
            tool_result=None,
            confidence=0.8,
            latency_ms=latency_ms
        )

    def _tool_discovery_phase(
        self,
        context: str,
        thinking: str,
        step_num: int
    ) -> ReasoningStep:
        """Dense retrieval-based tool discovery (DeepAgent's approach)"""
        start_time = time.time()

        # Create query embedding
        query_text = f"{context} {thinking}"

        if self.embedder:
            try:
                query_embedding = self.embedder.encode(query_text, convert_to_numpy=True)

                # Compute similarity to all tools
                tool_scores = []
                for tool_name, tool in self.tools.items():
                    if tool.embedding is not None:
                        # Cosine similarity
                        similarity = np.dot(query_embedding, tool.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(tool.embedding)
                        )

                        # Boost by success rate (learned from ToolPO)
                        adjusted_score = similarity * (0.7 + 0.3 * tool.success_rate)

                        tool_scores.append((tool_name, adjusted_score))

                # Rank tools
                tool_scores.sort(key=lambda x: x[1], reverse=True)

                # Top 3 candidates
                top_tools = [name for name, score in tool_scores[:3]]

                # Select best tool (in real impl, LLM would decide)
                selected_tool = top_tools[0] if top_tools else None

            except Exception as e:
                print(f"⚠️  Tool discovery failed: {e}")
                selected_tool = None
                top_tools = []

        else:
            # Fallback: keyword matching
            top_tools = self._keyword_tool_matching(query_text)
            selected_tool = top_tools[0] if top_tools else None

        latency_ms = int((time.time() - start_time) * 1000)

        discovery_content = f"Discovered tools: {', '.join(top_tools[:3]) if top_tools else 'None'}"

        return ReasoningStep(
            step_num=step_num,
            step_type="tool_discovery",
            content=discovery_content,
            tools_considered=top_tools,
            tool_selected=selected_tool,
            tool_result=None,
            confidence=0.75,
            latency_ms=latency_ms
        )

    def _keyword_tool_matching(self, query: str) -> List[str]:
        """Fallback keyword-based tool matching"""
        query_lower = query.lower()

        matches = []
        for tool_name, tool in self.tools.items():
            # Simple keyword overlap
            tool_text = f"{tool.name} {tool.description} {tool.category}".lower()

            # Count matching words
            query_words = set(query_lower.split())
            tool_words = set(tool_text.split())
            overlap = len(query_words & tool_words)

            if overlap > 0:
                matches.append((tool_name, overlap))

        # Sort by overlap
        matches.sort(key=lambda x: x[1], reverse=True)

        return [name for name, score in matches[:3]]

    def _tool_execution_phase(
        self,
        tool_name: str,
        context: str,
        step_num: int
    ) -> ReasoningStep:
        """Execute selected tool"""
        start_time = time.time()

        # Simulate tool execution (in real impl, would invoke MCP server)
        tool = self.tools.get(tool_name)

        if tool:
            tool_result = f"[Simulated result from {tool_name}]: Success. {tool.example_usage}"
            success = True

            # Update tool statistics (ToolPO-style learning)
            tool.use_count += 1

        else:
            tool_result = f"Error: Tool {tool_name} not found"
            success = False

        latency_ms = int((time.time() - start_time) * 1000)

        return ReasoningStep(
            step_num=step_num,
            step_type="tool_execution",
            content=f"Executed {tool_name}",
            tools_considered=[],
            tool_selected=tool_name,
            tool_result=tool_result,
            confidence=0.9 if success else 0.3,
            latency_ms=latency_ms,
            metadata={"success": success}
        )

    def _reflection_phase(
        self,
        context: str,
        steps: List[ReasoningStep],
        step_num: int
    ) -> ReasoningStep:
        """Periodic reflection on progress"""
        start_time = time.time()

        # Analyze progress
        tools_used = [s.tool_selected for s in steps if s.tool_selected]
        thinking_steps = [s for s in steps if s.step_type == "thinking"]

        reflection_content = (
            f"Reflection at step {step_num + 1}: "
            f"Completed {len(steps)} steps, used {len(set(tools_used))} unique tools. "
            f"Progress assessment: {'On track' if len(steps) < 15 else 'May need to refocus'}."
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return ReasoningStep(
            step_num=step_num,
            step_type="reflection",
            content=reflection_content,
            tools_considered=[],
            tool_selected=None,
            tool_result=None,
            confidence=0.7,
            latency_ms=latency_ms
        )

    def _is_task_complete(self, thinking: str) -> bool:
        """Check if task appears complete"""
        completion_indicators = [
            "complete", "done", "finished", "final answer",
            "solution is", "result is", "conclusion"
        ]

        thinking_lower = thinking.lower()
        return any(indicator in thinking_lower for indicator in completion_indicators)

    def _should_use_tool(self, thinking: str) -> bool:
        """Determine if tool use is beneficial"""
        tool_indicators = [
            "need to", "should search", "require", "look up",
            "find", "scan", "check", "verify", "retrieve"
        ]

        thinking_lower = thinking.lower()
        return any(indicator in thinking_lower for indicator in tool_indicators)

    def _fold_memory(self):
        """Autonomous memory folding (DeepAgent's approach)"""
        print("   📦 Folding memory...")

        # Compress working memory into episodic memory
        if len(self.working_memory) > 10:
            # Take oldest memories
            to_fold = self.working_memory[:5]

            # Create compressed representation
            compressed = self._compress_memory_block(to_fold)

            # Create fold
            fold = MemoryFold(
                fold_id=hashlib.md5(str(to_fold).encode()).hexdigest()[:16],
                fold_type="episodic",
                original_length=len(" ".join(to_fold)),
                compressed_length=len(compressed),
                content=compressed,
                timestamp=datetime.now(),
                importance_score=0.7
            )

            self.memory_folds.append(fold)

            # Remove from working memory
            self.working_memory = self.working_memory[5:]

            # Add reference
            self.working_memory.insert(0, f"[Memory fold #{len(self.memory_folds)}]")

            print(f"   ✅ Folded {fold.original_length} → {fold.compressed_length} chars")

    def _compress_memory_block(self, memory_items: List[str]) -> str:
        """Compress memory items into compact representation"""
        # Simple compression: keep key information
        combined = " | ".join(memory_items)

        # In real impl, would use LLM to summarize
        if len(combined) > 200:
            return combined[:200] + "... [continued]"

        return combined

    def _assess_quality(self, answer: str, prompt: str) -> float:
        """Assess quality of final answer"""
        if not answer:
            return 0.0

        # Simple heuristics (in real impl, would use LLM-as-judge)
        score = 0.5  # Base score

        if len(answer) > 50:
            score += 0.2  # Not too short

        if any(word in answer.lower() for word in ["because", "therefore", "reasoning"]):
            score += 0.2  # Has explanation

        if len(answer) < 2000:
            score += 0.1  # Not too verbose

        return min(score, 1.0)

    def _learn_from_trace(self, trace: ReasoningTrace):
        """Learn from reasoning trace (ToolPO-style)"""
        # Update tool success rates based on outcome
        if trace.success:
            for tool_name in trace.tools_used:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]

                    # Increase success rate (exponential moving average)
                    tool.success_rate = 0.9 * tool.success_rate + 0.1 * 1.0

                    # Update average latency
                    tool_latencies = [
                        s.latency_ms for s in trace.steps
                        if s.tool_selected == tool_name
                    ]
                    if tool_latencies:
                        avg_lat = sum(tool_latencies) / len(tool_latencies)
                        tool.avg_latency_ms = int(0.8 * tool.avg_latency_ms + 0.2 * avg_lat)

        else:
            # Decrease success rate for failed tasks
            for tool_name in trace.tools_used:
                if tool_name in self.tools:
                    tool = self.tools[tool_name]
                    tool.success_rate = 0.9 * tool.success_rate + 0.1 * 0.0

        # Save updated tool memory
        self._save_tool_memory()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            "tools": {
                "total_tools": len(self.tools),
                "by_category": self._count_by_category(),
                "most_used": self._get_most_used_tools(5),
                "best_performing": self._get_best_performing_tools(5)
            },
            "reasoning": {
                "total_traces": len(self.reasoning_traces),
                "avg_steps": np.mean([len(t.steps) for t in self.reasoning_traces]) if self.reasoning_traces else 0,
                "avg_latency_ms": np.mean([t.total_latency_ms for t in self.reasoning_traces]) if self.reasoning_traces else 0,
                "success_rate": np.mean([t.success for t in self.reasoning_traces]) if self.reasoning_traces else 0
            },
            "memory": {
                "working_memory_items": len(self.working_memory),
                "episodic_memory_items": len(self.episodic_memory),
                "total_folds": len(self.memory_folds),
                "avg_compression_ratio": np.mean([
                    f.compressed_length / f.original_length
                    for f in self.memory_folds
                ]) if self.memory_folds else 0
            }
        }

    def _count_by_category(self) -> Dict[str, int]:
        """Count tools by category"""
        counts = {}
        for tool in self.tools.values():
            counts[tool.category] = counts.get(tool.category, 0) + 1
        return counts

    def _get_most_used_tools(self, n: int) -> List[Dict[str, Any]]:
        """Get most used tools"""
        sorted_tools = sorted(
            self.tools.items(),
            key=lambda x: x[1].use_count,
            reverse=True
        )

        return [
            {"name": name, "use_count": tool.use_count}
            for name, tool in sorted_tools[:n]
        ]

    def _get_best_performing_tools(self, n: int) -> List[Dict[str, Any]]:
        """Get best performing tools"""
        sorted_tools = sorted(
            self.tools.items(),
            key=lambda x: x[1].success_rate,
            reverse=True
        )

        return [
            {"name": name, "success_rate": tool.success_rate}
            for name, tool in sorted_tools[:n]
            if tool.use_count > 0  # Only tools that have been used
        ]


def main():
    """Demo deep reasoning agent"""
    print("="*70)
    print("Deep Reasoning Agent - Demo")
    print("="*70)

    agent = DeepReasoningAgent()

    # Example task
    task = "Search the codebase for security vulnerabilities and explain the findings"

    trace = agent.reason(
        task_prompt=task,
        max_steps=10,
        thinking_budget=3,
        fold_threshold=8
    )

    print("\n📊 Reasoning Trace:")
    print(f"   Task: {trace.task_prompt}")
    print(f"   Steps: {len(trace.steps)}")
    print(f"   Tools used: {', '.join(trace.tools_used)}")
    print(f"   Latency: {trace.total_latency_ms}ms")
    print(f"   Success: {trace.success}")
    print(f"   Quality: {trace.quality_score:.2f}")

    print("\n📊 Agent Statistics:")
    stats = agent.get_statistics()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
