#!/usr/bin/env python3
"""
Novel Psychoactive Substance (NPS) Abuse Potential Analyzer CLI
PROACTIVE threat intelligence for emerging designer drugs

Usage:
    python3 nps_cli.py "classify mol_1 as nps"
    python3 nps_cli.py "predict abuse potential for mol_2"
    python3 nps_cli.py "comprehensive analysis of fentanyl analog"
    python3 nps_cli.py "batch screen 100 compounds for abuse risk"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.nps_abuse_potential_analyzer import NPSAbusePotentialAnalyzer

class NPSCLI:
    def __init__(self):
        self.analyzer = NPSAbusePotentialAnalyzer(verbose=True)

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Parse molecule (SMILES or name)
        if 'parse' in query_lower or 'load' in query_lower:
            smiles_match = re.search(r'["\']([^"\']+)["\']', query)
            if smiles_match:
                structure = smiles_match.group(1)
            else:
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

        # NPS classification
        elif 'classify' in query_lower or 'classification' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'classify',
                'mol_id': mol_id
            }

        # Abuse potential prediction
        elif 'abuse' in query_lower or 'recreational' in query_lower:
            mol_id = self._extract_mol_id(query)

            # Check if comprehensive mode requested
            comprehensive = 'comprehensive' in query_lower or 'full' in query_lower or '12' in query

            return {
                'action': 'abuse_potential',
                'mol_id': mol_id,
                'comprehensive': comprehensive
            }

        # Receptor binding prediction
        elif 'receptor' in query_lower or 'binding' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'receptor_binding',
                'mol_id': mol_id
            }

        # Antidote recommendations
        elif 'antidote' in query_lower or 'overdose' in query_lower:
            mol_id = self._extract_mol_id(query)
            return {
                'action': 'antidote',
                'mol_id': mol_id
            }

        # Batch screening
        elif 'batch' in query_lower or 'screen' in query_lower:
            # Extract number of compounds if specified
            count_match = re.search(r'(\d+)\s*(?:compounds?|molecules?)', query_lower)
            count = int(count_match.group(1)) if count_match else None

            # Extract molecule IDs
            mol_ids = re.findall(r'mol_\w+', query)
            if not mol_ids and not count:
                mol_ids = None

            return {
                'action': 'batch',
                'mol_ids': mol_ids,
                'count': count
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
        mol_match = re.search(r'(mol_\w+)', query)
        if mol_match:
            return mol_match.group(1)

        parts = query.split()
        for i, part in enumerate(parts):
            if part.lower() in ['for', 'of', 'molecule', 'compound', 'substance']:
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
                print('   Example: python3 nps_cli.py \'parse "CCN(CC)C(=O)C1CN(C)CCc2ccccc21" as Fentanyl\'')
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

                # Auto-classify as NPS
                print(f"\n   Running preliminary NPS classification...")
                nps_result = self.analyzer.classify_nps(mol_id=mol_id)
                if nps_result.get('success') and nps_result.get('is_nps'):
                    print(f"   ⚠️  WARNING: Classified as Novel Psychoactive Substance!")
                    print(f"      Class: {nps_result.get('nps_class', 'Unknown')}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'classify':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                print("   Example: python3 nps_cli.py 'classify mol_1'")
                return

            result = self.analyzer.classify_nps(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ NPS Classification for {parsed['mol_id']}")
                print(f"\n   Is NPS: {'⚠️  YES' if result['is_nps'] else '✅ NO'}")

                if result['is_nps']:
                    print(f"   NPS Class: {result['nps_class']}")
                    print(f"   Controlled Substance: {'YES' if result['controlled_substance'] else 'NO'}")
                    print(f"   Predicted DEA Schedule: {result['dea_schedule']}")

                    print(f"\n   Matched Patterns:")
                    for pattern in result['matched_patterns']:
                        print(f"      • {pattern}")

                    print(f"\n   Similarity to Known NPS:")
                    for nps, similarity in result['similarity_to_known_nps'].items():
                        print(f"      {nps}: {similarity:.1%}")

                    if result.get('regulatory_recommendations'):
                        print(f"\n   📋 Regulatory Recommendations:")
                        for rec in result['regulatory_recommendations']:
                            print(f"      • {rec}")
                else:
                    print(f"   ✅ Not classified as NPS")
                    print(f"   This substance does not match known NPS structural patterns")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'abuse_potential':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            comprehensive = parsed.get('comprehensive', False)

            if comprehensive:
                print(f"🔬 Starting COMPREHENSIVE abuse potential analysis...")
                print(f"   ⚠️  This may take up to 12 hours for thorough analysis")
                print(f"   Analyzing: {parsed['mol_id']}\n")

            result = self.analyzer.predict_abuse_potential(
                mol_id=parsed['mol_id'],
                comprehensive=comprehensive
            )

            if result.get('success'):
                print(f"✅ Abuse Potential Analysis Complete")
                print(f"\n{'='*70}")
                print(f"   SUBSTANCE: {parsed['mol_id']}")
                print(f"{'='*70}")

                print(f"\n   📊 NPS Classification:")
                print(f"      Class: {result['nps_classification']['nps_class']}")
                print(f"      Controlled: {result['nps_classification']['controlled_substance']}")
                print(f"      DEA Schedule: {result['nps_classification']['dea_schedule']}")

                print(f"\n   💊 Similarity to Known Drugs:")
                for drug, similarity in result['similarity_analysis']['top_matches'].items():
                    print(f"      {drug}: {similarity:.1%}")

                print(f"\n   🧠 Receptor Binding Prediction:")
                for receptor, data in result['receptor_binding'].items():
                    if data['binding_score'] > 0.3:
                        print(f"      {receptor}: {data['binding_score']:.2f} ({data['affinity']})")

                print(f"\n   ⚡ Reinforcement Mechanisms:")
                print(f"      Dopamine Release: {result['reinforcement_mechanisms']['dopamine_release']:.2f}/10")
                print(f"      Euphoria Potential: {result['reinforcement_mechanisms']['euphoria_potential']:.2f}/10")
                print(f"      Addiction Risk: {result['reinforcement_mechanisms']['addiction_risk']}")

                print(f"\n   ☠️  Risk Assessment:")
                print(f"      Neurotoxicity: {result['neurotoxicity_assessment']['neurotoxicity_score']:.2f}/10")
                print(f"      Lethality Risk: {result['lethality_assessment']['lethality_score']:.2f}/10")
                print(f"      LD50 Estimate: {result['lethality_assessment']['estimated_ld50']}")

                print(f"\n   💉 Antidote Recommendations:")
                if result['antidote_recommendations']['primary_antidotes']:
                    for antidote in result['antidote_recommendations']['primary_antidotes']:
                        print(f"      • {antidote}")
                else:
                    print(f"      ⚠️  No specific antidote identified")

                print(f"\n   🌐 Dark Web Proliferation Risk:")
                print(f"      Proliferation Score: {result['proliferation_prediction']['dark_web_score']:.2f}/10")
                print(f"      Likelihood: {result['proliferation_prediction']['proliferation_likelihood']}")

                print(f"\n   ⚠️  OVERALL ABUSE POTENTIAL SCORE: {result['abuse_potential_score']:.1f}/10")
                print(f"   Risk Category: {result['risk_category']}")

                if result.get('warnings'):
                    print(f"\n   🚨 Critical Warnings:")
                    for warning in result['warnings']:
                        print(f"      • {warning}")

                if result.get('recommendations'):
                    print(f"\n   📝 Recommendations:")
                    for i, rec in enumerate(result['recommendations'], 1):
                        print(f"      {i}. {rec}")

                print(f"\n{'='*70}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'receptor_binding':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            result = self.analyzer.predict_receptor_binding(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ Receptor Binding Prediction for {parsed['mol_id']}")
                print(f"\n   Predicted Receptor Interactions:")

                for receptor, data in result['receptor_predictions'].items():
                    print(f"\n   {receptor.upper()}:")
                    print(f"      Binding Score: {data['binding_score']:.2f}")
                    print(f"      Affinity: {data['affinity']}")
                    print(f"      Mechanism: {data['mechanism']}")
                    if data.get('effects'):
                        print(f"      Effects: {', '.join(data['effects'])}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'antidote':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID")
                return

            # First classify to determine antidote
            classify_result = self.analyzer.classify_nps(mol_id=parsed['mol_id'])

            if classify_result.get('success'):
                # Then get abuse potential for full antidote recommendations
                abuse_result = self.analyzer.predict_abuse_potential(
                    mol_id=parsed['mol_id'],
                    comprehensive=False
                )

                if abuse_result.get('success'):
                    antidotes = abuse_result['antidote_recommendations']

                    print(f"✅ Antidote Recommendations for {parsed['mol_id']}")
                    print(f"   NPS Class: {classify_result['nps_class']}")

                    if antidotes['primary_antidotes']:
                        print(f"\n   💉 Primary Antidotes:")
                        for antidote in antidotes['primary_antidotes']:
                            print(f"      • {antidote}")

                    if antidotes.get('supportive_care'):
                        print(f"\n   🏥 Supportive Care:")
                        for care in antidotes['supportive_care']:
                            print(f"      • {care}")

                    if antidotes.get('emergency_protocols'):
                        print(f"\n   🚨 Emergency Protocols:")
                        for protocol in antidotes['emergency_protocols']:
                            print(f"      • {protocol}")
                else:
                    print(f"❌ Error analyzing abuse potential: {abuse_result.get('error')}")
            else:
                print(f"❌ Error classifying NPS: {classify_result.get('error')}")

        elif action == 'batch':
            mol_ids = parsed.get('mol_ids')
            count = parsed.get('count')

            if not mol_ids and not count:
                print("❌ Please specify molecule IDs or compound count for batch screening")
                print("   Example: python3 nps_cli.py 'batch screen mol_1 mol_2 mol_3'")
                print("   Example: python3 nps_cli.py 'screen 100 compounds for abuse potential'")
                return

            if mol_ids:
                result = self.analyzer.batch_screening(
                    mol_ids=mol_ids,
                    output_dir=str(Path.home() / ".dsmil" / "nps_results")
                )

                if result.get('success'):
                    print(f"✅ Batch Screening Complete!")
                    print(f"   Total Screened: {result['total_screened']}")
                    print(f"   High Risk: {result['high_risk_count']}")
                    print(f"   Medium Risk: {result['medium_risk_count']}")
                    print(f"   Low Risk: {result['low_risk_count']}")

                    print(f"\n   Results saved to: {result['output_directory']}")

                    if result.get('high_risk_substances'):
                        print(f"\n   🚨 HIGH RISK SUBSTANCES:")
                        for substance in result['high_risk_substances'][:10]:
                            print(f"      {substance['mol_id']}: Score {substance['abuse_score']:.1f}/10 - {substance['nps_class']}")

                    print(f"\n   Processing Time: {result.get('processing_time', 'N/A')}")
                    print(f"   Average Time per Molecule: {result.get('avg_time_per_molecule', 'N/A')}")
                else:
                    print(f"❌ Error: {result.get('error')}")
            else:
                print(f"ℹ️  Batch screening of {count} compounds")
                print(f"   Note: Molecules must be loaded first using 'parse' command")
                print(f"   For large-scale screening (1M+ compounds), use the batch_screening() API directly")

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
            print(f"🔬 NPS Abuse Potential Analyzer Status")
            print(f"   Available: {status['available']}")
            print(f"   Loaded Molecules: {status['loaded_molecules']}")
            print(f"   Storage: {status['storage_path']}")

            print(f"\n   Known NPS Classes:")
            for nps_class in status['nps_classes']:
                print(f"      • {nps_class}")

            print(f"\n   Reference Drugs:")
            for drug in status['reference_drugs']:
                print(f"      • {drug}")

            print(f"\n   Analysis Capabilities:")
            for cap in status['capabilities']:
                print(f"      ✅ {cap}")

            print(f"\n   Receptor Systems Analyzed:")
            for receptor in status['receptor_systems']:
                print(f"      • {receptor}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
🔬 Novel Psychoactive Substance (NPS) Abuse Potential Analyzer

PROACTIVE threat intelligence for emerging designer drugs

USAGE:
    python3 nps_cli.py "your natural language command"

EXAMPLES:
    # Parse molecules
    python3 nps_cli.py 'parse "CCN(CC)C(=O)C1CN(C)CCc2ccccc21" as Fentanyl'
    python3 nps_cli.py 'parse "CC(C)CC(NC(=O)C(C)NC(=O)C1CCCN1)C(=O)NC" as Synthetic_Cannabinoid'

    # Classify as NPS
    python3 nps_cli.py "classify mol_1"
    python3 nps_cli.py "is mol_2 a novel psychoactive substance"

    # Predict abuse potential
    python3 nps_cli.py "predict abuse potential for mol_1"
    python3 nps_cli.py "comprehensive abuse analysis of mol_2"  # 12-hour mode

    # Receptor binding
    python3 nps_cli.py "predict receptor binding for mol_3"

    # Antidote recommendations
    python3 nps_cli.py "recommend antidote for mol_1"
    python3 nps_cli.py "overdose treatment for fentanyl analog"

    # Batch screening
    python3 nps_cli.py "batch screen mol_1 mol_2 mol_3 mol_4"
    python3 nps_cli.py "screen 1000 compounds for abuse potential"

    # Utility
    python3 nps_cli.py "list molecules"
    python3 nps_cli.py "status"

ANALYSIS FEATURES:
    ✅ NPS classification (20+ chemical classes)
    ✅ Abuse potential scoring (0-10 scale)
    ✅ Receptor binding prediction (6 neurotransmitter systems)
    ✅ Neurotoxicity assessment
    ✅ Lethality risk prediction
    ✅ Antidote recommendations
    ✅ Dark web proliferation prediction
    ✅ DEA scheduling recommendations
    ✅ Comprehensive 12-hour analysis mode
    ✅ Batch screening (1M+ compounds)

SUPPORTED NPS CLASSES:
    • Synthetic cannabinoids (JWH, AB-FUBINACA, etc.)
    • Synthetic cathinones (bath salts)
    • Fentanyl analogs
    • Benzodiazepine analogs
    • NBOMe compounds
    • 2C-x psychedelics
    • Tryptamines
    • Phenethylamines
    • Designer opioids
    • And 10+ more classes

RECEPTOR SYSTEMS ANALYZED:
    • Opioid receptors (μ, κ, δ)
    • Dopamine receptors
    • Serotonin receptors (5-HT)
    • NMDA/Glutamate receptors
    • GABA receptors
    • Cannabinoid receptors (CB1, CB2)

PROACTIVE CAPABILITIES:
    🔮 Predict abuse potential of NOT-YET-SYNTHESIZED substances
    🌐 Estimate dark web proliferation likelihood
    💉 Recommend antidotes BEFORE substances emerge
    📋 Generate regulatory recommendations (DEA scheduling)
    🚨 Identify high-risk analogs for early intervention

USE CASES:
    🔬 Pharmaceutical safety assessment
    🚔 Law enforcement threat intelligence
    🏥 Emergency medicine preparedness
    📊 Regulatory agency decision support
    🛡️  Proactive drug policy development

TEMPEST COMPLIANCE:
    • All operations local (RDKit computational chemistry)
    • No network communication required
    • Air-gapped deployment compatible
    • Suitable for classified law enforcement environments
    • EM emissions: Minimal (CPU-bound calculations only)

SECURITY & ETHICAL NOTES:
    ⚠️  This tool is for AUTHORIZED USE ONLY:
       • Law enforcement threat intelligence
       • Pharmaceutical safety research
       • Emergency medicine preparedness
       • Regulatory agency decision support

    ❌ NOT FOR:
       • Illicit drug synthesis guidance
       • Recreational drug design
       • Circumventing controlled substance laws

    📝 Results are PREDICTIONS, not clinical validation
    🧪 Requires wet-lab validation before regulatory action
    ⚖️  Designed to PREVENT harm, not enable it

PERFORMANCE:
    • Standard analysis: ~1-5 seconds per compound
    • Comprehensive (12-hour): Deep molecular dynamics simulation
    • Batch screening: ~1M compounds in 24 hours (parallelized)
        """)

def main():
    if len(sys.argv) < 2:
        cli = NPSCLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = NPSCLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
