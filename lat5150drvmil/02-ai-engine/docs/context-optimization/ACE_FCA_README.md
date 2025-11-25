# ACE-FCA: Advanced Context Engineering for Coding Agents

## Overview

This implementation integrates **ACE-FCA** (Advanced Context Engineering for Coding Agents) patterns into the DSMIL AI system, significantly improving the quality and reliability of AI-assisted coding tasks.

**Based on:** [HumanLayer's ACE-FCA Methodology](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)

## Key Principle

> **LLMs are stateless functions where context window quality is the only lever for output quality.**

Rather than waiting for smarter models, sophisticated context management unlocks production-grade AI coding today.

## Core Features

### 1. **Frequent Intentional Compaction**
- Maintains context utilization at **40-60%** for optimal reasoning quality
- Automatic compaction triggers at **75%** threshold
- Preserves critical information while removing noise

### 2. **Phase-Based Workflows**
Structured **Research → Plan → Implement → Verify** workflow:

- **Research Phase**: Understand codebase architecture and patterns
- **Planning Phase**: Create detailed implementation steps
- **Implementation Phase**: Execute plan phase-by-phase
- **Verification Phase**: Test and validate results

### 3. **Specialized Subagents**
Context-isolated subagents that return **compressed findings**:

- **ResearchAgent**: Codebase exploration (returns < 500 tokens)
- **PlannerAgent**: Implementation planning (returns < 600 tokens)
- **ImplementerAgent**: Code generation (returns < 500 tokens)
- **VerifierAgent**: Testing and validation (returns < 500 tokens)
- **SummarizerAgent**: Content compression (custom target)

### 4. **Human Review Checkpoints**
- Review at compaction boundaries (between phases)
- Approve/reject with feedback
- Critical for complex architectural decisions

### 5. **Context Quality Hierarchy**
Prioritizes by impact (worst to least bad):
1. **Incorrect information** ← WORST
2. **Missing information** ← BAD
3. **Excessive noise** ← LEAST BAD

## Architecture

```
┌─────────────────────────────────────────────────┐
│        UnifiedAIOrchestrator                    │
│  (with ACE-FCA Integration)                     │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐  ┌──────▼──────────┐
│ ACEContext   │  │ ACEWorkflow     │
│ Engine       │  │ Orchestrator    │
│              │  │                 │
│ • Compaction │  │ • Phase mgmt    │
│ • Token      │  │ • Review        │
│   tracking   │  │   checkpoints   │
│ • Priority   │  │ • Subagent      │
│   management │  │   coordination  │
└──────────────┘  └─────────┬───────┘
                            │
        ┌───────────────────┴────────────────────┐
        │                                        │
┌───────▼────────┐  ┌──────────────┐  ┌────────▼────────┐
│ ResearchAgent  │  │ PlannerAgent │  │ ImplementerAgent│
│ (context       │  │ (context     │  │ (context        │
│  isolated)     │  │  isolated)   │  │  isolated)      │
└────────────────┘  └──────────────┘  └─────────────────┘
```

## Files

### Core Modules

1. **`ace_context_engine.py`** (570 lines)
   - Context window management
   - Token estimation and tracking
   - Compaction strategies
   - Phase output management

2. **`ace_workflow_orchestrator.py`** (350 lines)
   - Phase-based workflow execution
   - Review checkpoint creation
   - Workflow task management
   - Progress tracking

3. **`ace_subagents.py`** (540 lines)
   - Specialized subagent implementations
   - Context isolation
   - Compressed output generation
   - File search and analysis

### Integration

4. **`unified_orchestrator.py`** (updated)
   - ACE-FCA integration
   - Workflow execution methods
   - Context management APIs
   - Status reporting

5. **`ai_tui_v2.py`** (updated)
   - ACE Workflow menu option
   - Interactive workflow interface
   - Phase output display
   - Review checkpoint handling

## Usage

### CLI Usage

```bash
# Check status (includes ACE-FCA info)
python3 unified_orchestrator.py status

# Execute workflow
python3 unified_orchestrator.py workflow "Add rate limiting to API" \
    --type feature \
    --complexity medium

# Use AI TUI
python3 ai_tui_v2.py
# Select: w → ACE Workflow
```

### Programmatic Usage

```python
from unified_orchestrator import UnifiedAIOrchestrator

# Initialize with ACE-FCA enabled
orchestrator = UnifiedAIOrchestrator(enable_ace=True)

# Execute workflow
result = orchestrator.execute_workflow(
    task_description="Add authentication to API endpoints",
    task_type="feature",
    complexity="medium",
    constraints=["Must maintain backward compatibility"],
    model_preference="quality_code"
)

if result['success']:
    print(f"Phases completed: {result['phases_completed']}")
    print(f"Research: {result['research_output']}")
    print(f"Plan: {result['plan_output']}")
    print(f"Implementation: {result['implementation_notes']}")
    print(f"Verification: {result['verification_results']}")
```

### Using Specialized Subagents

```python
# Research subagent (context isolated)
result = orchestrator.use_subagent('research', {
    'query': 'authentication',
    'search_paths': ['./src'],
    'file_patterns': ['*.py']
})
print(result['compressed_output'])  # < 500 tokens

# Planner subagent
result = orchestrator.use_subagent('planner', {
    'description': 'Add JWT authentication',
    'research_findings': research_output,
    'constraints': ['OAuth2 compatible']
})
print(result['compressed_output'])  # < 600 tokens

# Summarizer subagent
result = orchestrator.use_subagent('summarizer', {
    'content': large_text,
    'max_tokens': 200,
    'focus': 'key security findings'
})
print(result['compressed_output'])  # < 200 tokens
```

### Context Management

```python
# Get context statistics
stats = orchestrator.get_context_stats()
print(f"Tokens: {stats['total_tokens']}/{stats['max_tokens']}")
print(f"Utilization: {stats['utilization_percent']}")
print(f"In optimal range: {stats['in_optimal_range']}")
print(f"Compactions performed: {stats['compaction_count']}")

# Manual compaction
result = orchestrator.compact_context(target_tokens=4096)
print(f"Freed {result['tokens_freed']} tokens")
```

## Workflow Example

```
User: Add rate limiting to API endpoints

┌─────────────────────────────────────────────┐
│ RESEARCH PHASE                              │
│ Context: 15% → Searching codebase...        │
│ ✓ Found 12 relevant files                   │
│ ✓ Identified REST API structure             │
│ ✓ Current patterns: Flask, no rate limiting │
└─────────────────────────────────────────────┘
        ↓ Compaction (500 tokens) ↓
┌─────────────────────────────────────────────┐
│ REVIEW CHECKPOINT: Research                 │
│ Context: 45% ✓                              │
│ [Shows compressed research findings]        │
│ Approve? [Y/n/feedback]: y                  │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ PLANNING PHASE                              │
│ Context: 48% ✓                              │
│ Phase 1: Create rate_limiter.py             │
│ Phase 2: Add middleware integration         │
│ Phase 3: Add configuration                  │
└─────────────────────────────────────────────┘
        ↓ Compaction (600 tokens) ↓
┌─────────────────────────────────────────────┐
│ REVIEW CHECKPOINT: Plan                     │
│ [Shows implementation plan]                 │
│ Approve? [Y/n/feedback]: y                  │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ IMPLEMENTATION PHASE                        │
│ Context: 52% ✓                              │
│ ✓ Created rate_limiter.py                   │
│ ✓ Added middleware                          │
│ ✓ Added configuration                       │
└─────────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────────┐
│ VERIFICATION PHASE                          │
│ Context: 55% ✓                              │
│ ✓ Tests: 15/15 passed                       │
│ ✓ Syntax: All files valid                   │
│ ✓ Code quality: No issues                   │
└─────────────────────────────────────────────┘
```

## Benefits

### Proven Results (from ACE-FCA paper)
- **300k LOC Rust codebase**: Bug fix merged on first submission
- **35k LOC feature**: 7 hours (vs estimated 3-5 days for senior engineer)

### Key Improvements
1. **Context Quality**: Maintains 40-60% utilization for optimal reasoning
2. **Reduced Hallucinations**: Compressed, high-quality context prevents drift
3. **Human Oversight**: Review checkpoints at critical boundaries
4. **Scalability**: Works on codebases of any size
5. **Efficiency**: Subagents return compressed findings only

## Best Practices

### 1. **Engage Deeply with Reviews**
> "You must engage deeply with your task or this will not work."

Don't rubber-stamp reviews - provide meaningful feedback.

### 2. **Let Context Compact Naturally**
Don't fight compaction. Trust the 40-60% range.

### 3. **Use Appropriate Models**
- Research: `fast` or `code`
- Planning: `quality_code`
- Implementation: `uncensored_code` or `quality_code`
- Verification: `quality_code`

### 4. **Provide Clear Constraints**
Be explicit about:
- Backward compatibility requirements
- Performance constraints
- Security requirements
- Code style preferences

### 5. **Review Each Phase**
Don't skip review checkpoints. They're critical for:
- Catching incorrect assumptions early
- Providing domain expertise
- Adjusting direction before heavy implementation

## Configuration

### Context Window Size
```python
# Default: 8192 tokens
orchestrator = UnifiedAIOrchestrator(enable_ace=True)

# Custom size
from ace_context_engine import ACEContextEngine
ace = ACEContextEngine(max_tokens=16384)
```

### Compaction Thresholds
```python
ace = ACEContextEngine(
    max_tokens=8192,
    target_utilization_min=0.40,  # 40%
    target_utilization_max=0.60,  # 60%
    compaction_trigger=0.75       # 75%
)
```

### Disable Human Review
```python
from ace_workflow_orchestrator import ACEWorkflowOrchestrator

orchestrator = ACEWorkflowOrchestrator(
    ai_engine=engine,
    enable_human_review=False  # Automated mode
)
```

## Status Check

```bash
$ python3 unified_orchestrator.py status
{
  "ace_fca": {
    "available": true,
    "features": [
      "Context compaction (40-60% utilization)",
      "Phase-based workflows (Research→Plan→Implement→Verify)",
      "Specialized subagents with context isolation",
      "Human review checkpoints at compaction boundaries"
    ],
    "context_stats": {
      "total_tokens": 0,
      "max_tokens": 8192,
      "utilization": 0.0,
      "in_optimal_range": false,
      "should_compact": false
    }
  }
}
```

## Troubleshooting

### "ACE-FCA not available"
- Check that `ace_context_engine.py`, `ace_workflow_orchestrator.py`, and `ace_subagents.py` are in `02-ai-engine/`
- Ensure imports are working: `python3 -c "from ace_context_engine import ACEContextEngine"`

### Context not compacting
- Check utilization: `orchestrator.get_context_stats()`
- Verify threshold: Compaction triggers at 75% by default
- Manual trigger: `orchestrator.compact_context()`

### Subagent returns too much data
- Subagents should return < 500-600 tokens
- Check compression is working in `_compress_output()`
- Adjust max_tokens in subagent execution

## References

1. [ACE-FCA Methodology](https://github.com/humanlayer/advanced-context-engineering-for-coding-agents/blob/main/ace-fca.md)
2. [HumanLayer](https://github.com/humanlayer/humanlayer)
3. [Context Window Management Paper](https://arxiv.org/abs/2310.06825)

## License

This implementation follows the project's existing license.

## Contributing

When contributing to ACE-FCA modules:
1. Maintain context compaction at 40-60%
2. Keep subagent outputs compressed (< 600 tokens)
3. Document phase transitions
4. Test with large codebases (> 100k LOC)

## Future Enhancements

- [ ] Semantic RAG instead of token-based
- [ ] Multi-agent coordination for parallel subagents
- [ ] Persistent context across sessions
- [ ] Advanced compaction strategies (LLM-based summarization)
- [ ] Workflow templates for common tasks
- [ ] Integration with version control for automated commits
