#!/usr/bin/env python3
"""
NMDA Agonist Antidepressant Analysis CLI - Natural Language Interface
Analyze NMDA receptor modulators for antidepressant properties

Usage:
    python3 nmda_cli.py "analyze CCO for nmda activity"
    python3 nmda_cli.py "compare mol_1 with ketamine"
    python3 nmda_cli.py "comprehensive analysis of esketamine analog"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.nmda_agonist_analyzer import NMDAAgonistAnalyzer

class NMDACLI:
    def __init__(self):
        self.analyzer = NMDAAgonistAnalyzer()

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Parse molecule (SMILES or name)
        if 'parse' in query_lower or 'load' in query_lower:
            # Extract SMILES or molecule name
            smiles_match = re.search(r'["\']([^"\']+)["\']', query)
            if smiles_match:
                structure = smiles_match.group(1)
            else:
                # Look for common patterns
                parts = query.split()
                for i, part in enumerate(parts):
                    if part.lower() in ['parse', 'load', 'molecule']:
                        if i + 1 < len(parts):
                            structure = parts[i + 1]
                            break
                else:
                    structure = None

            name_match = re.search(r'as ([a-zA-Z0-9_-]+)', query)
            name = name_match.group(1) if name_match else None

            return {
                'action': 'parse',
                'structure': structure,
                'name': name
            }

        # NMDA activity analysis
        elif 'nmda' in query_lower and ('analyze' in query_lower or 'activity' in query_lower):
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'nmda_activity',
                'mol_id': mol_id
            }

        # Blood-Brain Barrier prediction
        elif 'bbb' in query_lower or 'blood brain barrier' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'bbb_prediction',
                'mol_id': mol_id
            }

        # Compare with known antidepressants
        elif 'compare' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'compare_antidepressants',
                'mol_id': mol_id
            }

        # Comprehensive analysis
        elif 'comprehensive' in query_lower or 'full' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'comprehensive',
                'mol_id': mol_id
            }

        # Batch analysis
        elif 'batch' in query_lower or 'screen' in query_lower:
            # Extract molecule IDs
            mol_ids = re.findall(r'mol_\w+', query)
            if not mol_ids:
                mol_ids = None

            return {
                'action': 'batch',
                'mol_ids': mol_ids
            }

        # List molecules
        elif 'list' in query_lower or 'show' in query_lower:
            return {'action': 'list'}

        # Status
        elif 'status' in query_lower or 'info' in query_lower:
            return {'action': 'status'}

        else:
            return {'action': 'help'}

    def _extract_mol_id(self, query: str) -> str:
        """Extract molecule ID from query"""
        # Look for mol_xxx pattern
        mol_match = re.search(r'(mol_\w+)', query)
        if mol_match:
            return mol_match.group(1)

        # Look for specific keywords followed by molecule reference
        parts = query.split()
        for i, part in enumerate(parts):
            if part.lower() in ['for', 'of', 'molecule', 'compound']:
                if i + 1 < len(parts):
                    return parts[i + 1]

        return None

    def execute(self, query: str):
        """Execute natural language command"""
        parsed = self.parse_command(query)
        action = parsed.get('action')

        if action == 'parse':
            if not parsed.get('structure'):
                print("❌ Please specify a molecule structure (SMILES)")
                print("   Example: python3 nmda_cli.py 'parse \"CC(=O)C(c1ccccc1Cl)N(C)C\" as Ketamine'")
                return

            result = self.analyzer.rdkit_agent.parse_molecule(
                structure=parsed['structure'],
                format='smiles',
                name=parsed.get('name')
            )

            if result.get('success'):
                mol_id = result['mol_id']
                print(f"✅ Molecule parsed successfully!")
                print(f"   Molecule ID: {mol_id}")
                print(f"   Formula: {result.get('formula', 'N/A')}")
                print(f"   Molecular Weight: {result.get('mw', 'N/A'):.2f}")
                print(f"   SMILES: {result.get('smiles', 'N/A')}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'nmda_activity':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                print("   Example: python3 nmda_cli.py 'analyze nmda activity for mol_1'")
                return

            result = self.analyzer.analyze_nmda_activity(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ NMDA Activity Analysis for {parsed['mol_id']}")
                print(f"\n   Structural Similarity:")
                print(f"      PCP scaffold: {result['structural_similarity']['pcp_scaffold']:.2%}")
                print(f"      Ketamine scaffold: {result['structural_similarity']['ketamine_scaffold']:.2%}")
                print(f"      Memantine scaffold: {result['structural_similarity']['memantine_scaffold']:.2%}")

                print(f"\n   Activity Prediction:")
                print(f"      NMDA Activity Score: {result['nmda_activity_score']:.2f}/10")
                print(f"      Likely Mechanism: {result['likely_mechanism']}")

                print(f"\n   BBB Penetration:")
                print(f"      BBB Score: {result['bbb_penetration']['bbb_score']:.2f}")
                print(f"      Prediction: {result['bbb_penetration']['prediction']}")
                print(f"      Confidence: {result['bbb_penetration']['confidence']}")

                if result.get('warnings'):
                    print(f"\n   ⚠️  Warnings:")
                    for warning in result['warnings']:
                        print(f"      • {warning}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'bbb_prediction':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            result = self.analyzer.predict_bbb_penetration(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ Blood-Brain Barrier Prediction for {parsed['mol_id']}")
                print(f"   BBB Score: {result['bbb_score']:.2f}")
                print(f"   Prediction: {result['prediction']}")
                print(f"   Confidence: {result['confidence']}")
                print(f"\n   Contributing Factors:")
                print(f"      LogP: {result['factors']['logp']:.2f} (optimal: 1-3)")
                print(f"      Molecular Weight: {result['factors']['mw']:.1f} (optimal: <450)")
                print(f"      H-Bond Donors: {result['factors']['hbd']} (optimal: ≤3)")
                print(f"      H-Bond Acceptors: {result['factors']['hba']} (optimal: ≤7)")
                print(f"      TPSA: {result['factors']['tpsa']:.1f} (optimal: <90)")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'compare_antidepressants':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            result = self.analyzer.compare_with_known_antidepressants(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ Comparison with Known NMDA Antidepressants")
                print(f"\n   Similarity Scores:")

                for drug, data in result['similarity_scores'].items():
                    print(f"\n   {drug.upper()}:")
                    print(f"      Structural Similarity: {data['structural_similarity']:.2%}")
                    print(f"      Property Similarity: {data['property_similarity']:.2%}")
                    print(f"      Overall Score: {data['overall_similarity']:.2%}")

                print(f"\n   Most Similar Drug: {result['most_similar_drug']}")
                print(f"   Novelty Score: {result['novelty_score']:.2f}/10")

                if result.get('recommendations'):
                    print(f"\n   💡 Recommendations:")
                    for rec in result['recommendations']:
                        print(f"      • {rec}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'comprehensive':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            result = self.analyzer.comprehensive_analysis(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ Comprehensive NMDA Antidepressant Analysis")
                print(f"\n{'='*70}")
                print(f"   MOLECULE: {parsed['mol_id']}")
                print(f"{'='*70}")

                print(f"\n   📊 NMDA Activity:")
                print(f"      Activity Score: {result['nmda_activity']['nmda_activity_score']:.2f}/10")
                print(f"      Mechanism: {result['nmda_activity']['likely_mechanism']}")

                print(f"\n   🧠 Blood-Brain Barrier:")
                print(f"      BBB Score: {result['bbb_penetration']['bbb_score']:.2f}")
                print(f"      Prediction: {result['bbb_penetration']['prediction']}")

                print(f"\n   💊 Comparison with Known Antidepressants:")
                for drug, similarity in result['comparison']['similarity_scores'].items():
                    print(f"      {drug}: {similarity['overall_similarity']:.1%} similar")

                print(f"\n   🆕 Novelty Score: {result['comparison']['novelty_score']:.2f}/10")

                print(f"\n   💡 Overall Assessment:")
                print(f"      Antidepressant Potential: {result['overall_assessment']['potential_score']:.2f}/10")
                print(f"      Risk Level: {result['overall_assessment']['risk_level']}")

                if result['overall_assessment'].get('recommendations'):
                    print(f"\n   📝 Recommendations:")
                    for i, rec in enumerate(result['overall_assessment']['recommendations'], 1):
                        print(f"      {i}. {rec}")

                if result['overall_assessment'].get('warnings'):
                    print(f"\n   ⚠️  Warnings:")
                    for warning in result['overall_assessment']['warnings']:
                        print(f"      • {warning}")

                print(f"\n{'='*70}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'batch':
            if not parsed.get('mol_ids'):
                print("❌ Please specify molecule IDs for batch analysis")
                print("   Example: python3 nmda_cli.py 'batch analyze mol_1 mol_2 mol_3'")
                return

            result = self.analyzer.batch_analysis(
                mol_ids=parsed['mol_ids'],
                output_dir=str(Path.home() / ".dsmil" / "nmda_results")
            )

            if result.get('success'):
                print(f"✅ Batch Analysis Complete!")
                print(f"   Molecules Analyzed: {result['total_molecules']}")
                print(f"   Successful: {result['successful']}")
                print(f"   Failed: {result['failed']}")
                print(f"\n   Results saved to: {result['output_directory']}")

                if result.get('summary'):
                    print(f"\n   Top Candidates:")
                    for i, candidate in enumerate(result['summary'][:5], 1):
                        print(f"      {i}. {candidate['mol_id']}: Score {candidate['score']:.2f}/10")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'list':
            molecules = self.analyzer.rdkit_agent.molecules
            if molecules:
                print(f"📋 Loaded Molecules ({len(molecules)}):")
                for mol_id, data in molecules.items():
                    print(f"\n   {mol_id}:")
                    print(f"      Name: {data.get('name', 'Unnamed')}")
                    print(f"      Formula: {data.get('formula', 'N/A')}")
                    print(f"      MW: {data.get('mw', 'N/A')}")
            else:
                print("📋 No molecules loaded")
                print("   Use 'parse' command to load molecules")

        elif action == 'status':
            status = self.analyzer.get_status()
            print(f"🧪 NMDA Agonist Analyzer Status")
            print(f"   Available: {status['available']}")
            print(f"   Loaded Molecules: {status['loaded_molecules']}")
            print(f"   Storage: {status['storage_path']}")

            print(f"\n   Known NMDA Antidepressants:")
            for drug in status['known_antidepressants']:
                print(f"      • {drug}")

            print(f"\n   Analysis Capabilities:")
            for cap in status['capabilities']:
                print(f"      ✅ {cap}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
🧪 NMDA Agonist Antidepressant Analysis CLI - Natural Language Interface

Analyze NMDA receptor modulators for antidepressant properties

USAGE:
    python3 nmda_cli.py "your natural language command"

EXAMPLES:
    # Parse molecules
    python3 nmda_cli.py 'parse "CC(=O)C(c1ccccc1Cl)N(C)C" as Ketamine'
    python3 nmda_cli.py 'parse "CN1C2CCC1C(C2)OC(=O)C(CO)c3ccccc3" as Esketamine'

    # Analyze NMDA activity
    python3 nmda_cli.py "analyze nmda activity for mol_1"
    python3 nmda_cli.py "check bbb penetration for mol_2"

    # Compare with known antidepressants
    python3 nmda_cli.py "compare mol_1 with known antidepressants"
    python3 nmda_cli.py "comprehensive analysis of mol_3"

    # Batch analysis
    python3 nmda_cli.py "batch analyze mol_1 mol_2 mol_3"

    # Utility
    python3 nmda_cli.py "list molecules"
    python3 nmda_cli.py "status"

ANALYSIS FEATURES:
    • NMDA receptor activity prediction (0-10 score)
    • Blood-Brain Barrier (BBB) penetration assessment
    • Structural similarity to ketamine/esketamine/memantine
    • Comparison with known NMDA antidepressants
    • Safety warnings (neurotoxicity, addiction potential)
    • Novelty scoring for novel compound assessment

KNOWN REFERENCE COMPOUNDS:
    • Ketamine (Ketalar) - NMDA antagonist, rapid-acting antidepressant
    • Esketamine (Spravato) - S-enantiomer of ketamine, FDA-approved
    • Memantine (Namenda) - NMDA antagonist, used in Alzheimer's disease

TEMPEST COMPLIANCE:
    • All operations local (RDKit computational chemistry)
    • No network communication required
    • Air-gapped deployment compatible
    • Suitable for classified pharmaceutical research
    • EM emissions: Minimal (CPU-bound calculations only)

SECURITY NOTES:
    • This tool is for RESEARCH PURPOSES ONLY
    • Designed for authorized pharmaceutical research
    • NOT for illicit drug synthesis guidance
    • Results are PREDICTIONS, not clinical validation
    • Requires wet-lab validation before any human use
        """)

def main():
    if len(sys.argv) < 2:
        cli = NMDACLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = NMDACLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
