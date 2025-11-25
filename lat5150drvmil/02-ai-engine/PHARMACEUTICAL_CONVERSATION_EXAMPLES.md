# Pharmaceutical Conversational AI - Example Workflows

**Date**: 2025-11-16
**Version**: 1.0.0
**Feature**: Multi-turn context-aware pharmaceutical research

---

## Overview

The Pharmaceutical Conversational AI enables natural language, multi-turn pharmaceutical research with:
- **Context awareness**: Remembers compounds analyzed in the session
- **Smart suggestions**: Recommends next analysis steps based on what's been done
- **TEMPEST compliance**: Automatically enforces security levels
- **Compound library**: Tracks all analyzed compounds within session

---

## Example 1: Fentanyl Safety Assessment

### Multi-Turn Conversation

```
User: "Analyze fentanyl for safety"

AI: ✅ Screened Fentanyl
    ⚠️  Novel Psychoactive Substance detected: synthetic_opioid
    ✅ Passes Lipinski Rule of Five (drug-like)

    💡 Suggested next steps:
       1. Predict ADMET properties with Intel AI
       2. Assess abuse potential (NPS classification)
       3. Generate comprehensive safety profile

User: "What's the abuse potential?"

AI: 🚨 HIGH abuse potential (score: 9.2/10)

    💡 Suggested next steps:
       1. Perform molecular docking to target receptors
       2. Generate comprehensive safety profile
       3. Analyze Blood-Brain Barrier penetration

User: "Show me the binding affinity to mu-opioid receptor"

AI: ✅ Molecular docking complete
    🎯 Strong binding detected (-11.8 kcal/mol)

    Docking Results:
      MOR: -11.8 kcal/mol (Ki: 2.1 nM)
      DOR: -9.4 kcal/mol (Ki: 130 nM)
      KOR: -8.7 kcal/mol (Ki: 420 nM)

    💡 Suggested next steps:
       1. Generate comprehensive safety profile
       2. Analyze Blood-Brain Barrier penetration
       3. Run patient simulation (100k virtual patients)

User: "Give me the full safety profile"

AI: ❌ Significant safety concerns (score: 2.3/10)
    ⚠️  4 warning(s) identified

    Warnings:
    - Extremely high abuse potential (9.2/10)
    - Strong MOR binding (overdose risk)
    - Respiratory depression
    - Rapid tolerance development

    💡 Suggested next steps:
       1. Upgrade to TEMPEST Level 3 for patient simulation and regulatory tools
```

---

## Example 2: Ketamine Antidepressant Research

### Context-Aware Workflow

```
User: "Screen ketamine as an antidepressant"

AI: ✅ Screened Ketamine
    ✅ Passes Lipinski Rule of Five (drug-like)

    💡 Suggested next steps:
       1. Predict ADMET properties with Intel AI
       2. Analyze Blood-Brain Barrier penetration
       3. Perform molecular docking to target receptors

User: "Analyze NMDA receptor activity"

AI: ✅ NMDA antidepressant analysis complete
    🎯 Strong NMDA activity (8.4/10)

    BBB Penetration: HIGH
    Similarity to esketamine: 0.92
    Rapid-acting potential: YES

    💡 Suggested next steps:
       1. Predict ADMET properties with Intel AI
       2. Perform molecular docking to target receptors
       3. Generate comprehensive safety profile

User: "What about BBB penetration?"

AI: ✅ ADMET analysis complete
    🧠 High BBB penetration - CNS active
    💊 Good oral bioavailability (78%)

    Bioavailability: 78%
    BBB: HIGH
    Half-life: 2-3 hours

    💡 Suggested next steps:
       1. Perform molecular docking to target receptors
       2. Generate comprehensive safety profile
       3. Run patient simulation (100k virtual patients)

User: "Dock it to NMDA receptor"

AI: ✅ Molecular docking complete
    🎯 Strong binding detected (-10.2 kcal/mol)

    Docking Results:
      NMDA: -10.2 kcal/mol (Ki: 31 nM)
      5HT2A: -8.9 kcal/mol (Ki: 310 nM)
      MOR: -7.2 kcal/mol (Ki: 5200 nM)

    💡 Suggested next steps:
       1. Generate comprehensive safety profile
       2. Compare with known pharmaceutical compounds
       3. Run patient simulation (100k virtual patients)

User: "Session status"

AI: 📊 Session Status:
       • 1 compound(s) analyzed
       • 4 conversation turn(s)
       • TEMPEST Level 1
       • Current: Ketamine
       • Analyses: screen, nmda, admet, dock
```

---

## Example 3: Novel Compound Discovery

### Exploring Unknown SMILES

```
User: "Analyze this compound: CC(=O)C(c1ccccc1Cl)N(C)C"

AI: ✅ Screened Compound
    ⚠️  Novel Psychoactive Substance detected: nmda_antagonist
    ✅ Passes Lipinski Rule of Five (drug-like)

    Molecular Properties:
      MW: 237.73 g/mol
      LogP: 2.85
      TPSA: 23.47 Ų

    💡 Suggested next steps:
       1. Predict ADMET properties with Intel AI
       2. Analyze Blood-Brain Barrier penetration
       3. Assess abuse potential (NPS classification)

User: "Is this safe?"

AI: ⚠️  MODERATE safety concerns (score: 5.8/10)
    ⚠️  2 warning(s) identified

    Warnings:
    - Dissociative properties (NMDA antagonist)
    - Potential for psychological dependence

    💡 Suggested next steps:
       1. Perform molecular docking to target receptors
       2. Assess abuse potential (NPS classification)
       3. Compare with known pharmaceutical compounds

User: "Compare with known antidepressants"

AI: ✅ Comparison complete

    Structural Similarity:
    - Ketamine: 1.00 (identical)
    - Esketamine: 0.98 (S-enantiomer)
    - Memantine: 0.42

    This compound IS ketamine, an FDA-approved treatment for
    treatment-resistant depression (TRD).

    💡 Suggested next steps:
       1. Run patient simulation (100k virtual patients)
       2. Generate regulatory submission dossier
```

---

## Example 4: ZEROPAIN Protocol Simulation

### Pain Management Research

```
User: "Tell me about ZEROPAIN pain protocols"

AI: ✅ ZEROPAIN pain management protocols integrated

    Capabilities:
    - Opioid protocol optimization
    - Patient simulation (100k patients)
    - PK/PD modeling
    - Zero-tolerance therapy design

    Info: Use patient simulation with multi-compound protocols
          for zero-tolerance therapy

    💡 Suggested next steps:
       1. Provide protocol specification for simulation
       2. Screen individual compounds first

User: "Simulate SR-17018 protocol"

AI: ❌ Patient simulation requires protocol specification

    💡 Suggestion: Use API or provide full protocol details:
       - compounds: ["SR-17018", "SR-14968", "Oxycodone"]
       - doses: [16.17, 25.31, 5.07]
       - frequencies: [2, 1, 4]  # BID, QD, Q6H
       - duration: 90 days

    Note: Patient simulation requires TEMPEST Level 3 (CLASSIFIED)
```

---

## Example 5: Interactive Session Management

### Session Commands

```
User: "status"

AI: 📊 Session Status:
       • 3 compound(s) analyzed
       • 12 conversation turn(s)
       • TEMPEST Level 2
       • Current: Fentanyl
       • Analyses: screen, dock, safety, abuse

User: "library"

AI: 📚 Compound Library (3):

    1. Fentanyl (mol_001)
       CCN(CC)C(=O)C1CN(C)CCc2ccccc21
       Analyses: screen, dock, safety, abuse

    2. Ketamine (mol_002)
       CC(=O)C(c1ccccc1Cl)N(C)C
       Analyses: screen, nmda, admet, dock

    3. MDMA (mol_003)
       CC(CC1=CC2=C(C=C1)OCO2)NC
       Analyses: screen, abuse

User: "level 2"

AI: ✅ TEMPEST level set to 2

    Now available:
    - Molecular docking
    - Comprehensive safety profiles
    - Full abuse potential analysis
    - Audit logging enabled

User: "What can I do at Level 3?"

AI: 🔒 TEMPEST Level 3 (CLASSIFIED) provides:

    - Patient simulation (100,000 virtual patients)
    - Protocol optimization
    - Regulatory submission dossiers
    - Comprehensive 12-hour abuse analysis
    - Proactive threat intelligence
    - Air-gap compatible operation

    Requires: Government authorization
    Rate limit: 100 requests/day
```

---

## Example 6: Cross-Platform Integration

### Combining Multiple Analyses

```
User: "Comprehensive analysis of cocaine"

AI: ✅ Screened Cocaine
    ⚠️  Novel Psychoactive Substance detected: tropane_alkaloid
    ✅ Passes Lipinski Rule of Five (drug-like)

    💡 Suggested next steps:
       1. Predict ADMET properties with Intel AI
       2. Assess abuse potential (NPS classification)
       3. Generate comprehensive safety profile

User: "Full safety and abuse assessment"

AI: 🚨 HIGH abuse potential (score: 8.7/10)
    ❌ Significant safety concerns (score: 3.1/10)

    NPS Classification: tropane_alkaloid

    Receptor Binding:
    - Dopamine transporter (DAT): Very high
    - Norepinephrine transporter: High
    - Serotonin transporter: Moderate

    Warnings:
    - Extreme addiction liability
    - Cardiotoxicity (hERG binding)
    - Neurotoxicity (dopaminergic depletion)
    - Hepatotoxicity

    Suggested Antidotes:
    - Benzodiazepines (for agitation)
    - Alpha/beta blockers (for cardiovascular effects)

    💡 Suggested next steps:
       1. Perform molecular docking to target receptors
       2. Compare with pharmaceutical stimulants
       3. Generate regulatory submission dossier

User: "How does this compare to therapeutic stimulants?"

AI: ✅ Comparison complete

    Cocaine vs Therapeutic Stimulants:

    Structural Similarity:
    - Methylphenidate (Ritalin): 0.38
    - Amphetamine (Adderall): 0.29
    - Modafinil (Provigil): 0.21

    Key Differences:
    - Cocaine: Tropane alkaloid, DAT/NET/SERT inhibitor
    - Methylphenidate: Piperidine, selective DAT inhibitor
    - Amphetamine: Phenethylamine, releasing agent

    Abuse Potential:
    - Cocaine: 8.7/10 (extreme)
    - Methylphenidate: 4.2/10 (moderate)
    - Amphetamine: 5.8/10 (moderate-high)

    Therapeutic Index:
    - Cocaine: 1.2 (narrow)
    - Methylphenidate: 8.5 (wide)
    - Amphetamine: 5.3 (moderate)
```

---

## CLI Usage Examples

### Single Query Mode

```bash
# Basic screening
python3 pharmaceutical_cli.py "screen fentanyl"

# ADMET analysis
python3 pharmaceutical_cli.py "analyze ADMET properties of ketamine"

# Docking
python3 pharmaceutical_cli.py "dock cocaine to dopamine transporter"

# Safety assessment
python3 pharmaceutical_cli.py "safety profile of MDMA"
```

### Interactive Conversation Mode

```bash
# Start interactive session
python3 pharmaceutical_cli.py --conversation

# With TEMPEST level
python3 pharmaceutical_cli.py --conversation --level 2

# With session persistence
python3 pharmaceutical_cli.py --conversation --session my_research.json

# Resume session
python3 pharmaceutical_cli.py --conversation --session my_research.json
```

---

## Advanced Features

### Context Persistence

Sessions are automatically saved when using `--session` flag:

```json
{
  "history": [
    {
      "timestamp": "2025-11-16T14:30:00",
      "query": "screen fentanyl",
      "task": {...},
      "result": {...}
    }
  ],
  "compounds": {
    "CCN(CC)C(=O)C1CN(C)CCc2ccccc21": {
      "name": "Fentanyl",
      "mol_id": "mol_001",
      "analyses_performed": ["screen", "dock", "safety"],
      "results": {...}
    }
  },
  "current_compound": "CCN(CC)C(=O)C1CN(C)CCc2ccccc21",
  "session_start": "2025-11-16T14:25:00",
  "tempest_level": 2
}
```

### Smart Suggestions Engine

The conversational agent provides context-aware suggestions based on:
1. **Completed analyses**: Avoids redundant work
2. **TEMPEST level**: Only suggests accessible capabilities
3. **Workflow logic**: Recommends sensible next steps
4. **Compound type**: Tailors suggestions to compound class

---

## Integration with Existing NLI

### Unified Orchestrator Integration

```python
from unified_orchestrator import UnifiedAIOrchestrator

orchestrator = UnifiedAIOrchestrator()

# Natural language pharmaceutical query
response = orchestrator.query("screen fentanyl for safety")

# Response includes:
# - conversational_response: Human-friendly explanation
# - conversation_turn: Turn number in session
# - suggestions: Smart next-step recommendations
# - session_id: Unique session identifier
```

### Smart Router Detection

Pharmaceutical queries are automatically detected:

```python
"screen fentanyl" → pharmaceutical backend → conversational mode
"dock ketamine to NMDA" → pharmaceutical backend → conversational mode
"ADMET of cocaine" → pharmaceutical backend → conversational mode
```

---

## TEMPEST Security in Conversations

### Level-Based Access Control

```
Level 0 (PUBLIC):
- Basic screening
- Molecular properties
- Drug-likeness

Level 1 (RESTRICTED):
- ADMET prediction
- BBB analysis
- Therapeutic classification

Level 2 (CONTROLLED):
- Molecular docking
- Safety profiles
- Abuse potential
- Audit logging

Level 3 (CLASSIFIED):
- Patient simulation
- Regulatory dossiers
- Comprehensive abuse analysis
- Protocol optimization
```

### Automatic Enforcement

The conversational agent automatically:
- Checks TEMPEST level before executing operations
- Provides upgrade suggestions when needed
- Logs Level 2+ operations
- Filters results based on clearance

---

## Performance Characteristics

### Response Times

- **Intent detection**: < 1 ms
- **Context retrieval**: < 5 ms
- **Suggestion generation**: < 10 ms
- **Total overhead**: < 20 ms

Plus underlying pharmaceutical analysis time.

### Memory Footprint

- **Per compound context**: ~5 KB
- **Per conversation turn**: ~2 KB
- **Session with 100 compounds**: ~700 KB

---

## Best Practices

### Effective Conversations

1. **Start with screening**: Always begin with compound screening
2. **Progressive analysis**: Follow suggested next steps
3. **Use context**: Reference "it", "this compound", "same"
4. **Check status**: Use "status" command to track progress
5. **Save sessions**: Use `--session` for persistence

### TEMPEST Compliance

1. **Know your level**: Use "level" command to check/set
2. **Follow suggestions**: System recommends accessible operations
3. **Upgrade intentionally**: Request higher levels when needed
4. **Respect limits**: Honor rate limiting per level

### Session Management

1. **Name sessions**: Use descriptive session filenames
2. **Save frequently**: Auto-saved after each turn with `--session`
3. **Review history**: Check conversation_history for audit trail
4. **Clean up**: Remove old sessions to save disk space

---

## Troubleshooting

### Common Issues

**"No compound specified"**
- Solution: Provide SMILES or recognized compound name
- Example: "screen fentanyl" or "screen CCN(CC)C(=O)..."

**"Requires TEMPEST Level X"**
- Solution: Upgrade security level with "level X" command
- Note: Higher levels require appropriate API keys

**"Patient simulation requires protocol specification"**
- Solution: Use API endpoint with full protocol dict
- CLI supports single-compound analysis only

---

## Next Steps

After mastering conversational pharmaceutical research:

1. **API Integration**: Use FastAPI endpoints for programmatic access
2. **Web Dashboard**: Visual interface at pharmaceutical_dashboard.html
3. **Batch Processing**: Process compound libraries via API
4. **Custom Workflows**: Build specialized analysis pipelines
5. **Integration**: Connect to LIMS, ELN, or research databases

---

**Status**: Conversational AI Integration Complete ✅
**Date**: 2025-11-16
**Version**: 1.0.0
