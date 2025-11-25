#!/usr/bin/env python3
"""
RDKit Cheminformatics CLI - Natural Language Interface
Molecular analysis, drug discovery, chemical fingerprinting

Usage:
    python3 rdkit_cli.py "parse molecule CCO as Ethanol"
    python3 rdkit_cli.py "calculate drug likeness for mol_1"
    python3 rdkit_cli.py "generate fingerprint for mol_1"
    python3 rdkit_cli.py "find molecules similar to mol_1"
"""

import sys
import json
import re
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sub_agents.rdkit_wrapper import RDKitAgent

class RDKitCLI:
    def __init__(self):
        self.agent = RDKitAgent()

    def parse_command(self, query: str) -> dict:
        """Parse natural language command into action and parameters"""
        query_lower = query.lower()

        # Parse molecule patterns
        if 'parse' in query_lower or 'load molecule' in query_lower or 'analyze molecule' in query_lower:
            # Extract SMILES or structure
            # Look for quoted strings or common SMILES patterns
            structure_match = re.search(r'(?:smiles |structure |molecule )([A-Za-z0-9()=\[\]@#+\-]+)', query)
            if not structure_match:
                structure_match = re.search(r'"([^"]+)"', query)

            if structure_match:
                structure = structure_match.group(1)

                # Extract format
                format_type = 'smiles'  # default
                if 'inchi' in query_lower:
                    format_type = 'inchi'
                elif 'mol' in query_lower or 'sdf' in query_lower:
                    format_type = 'mol'

                # Extract name
                name_match = re.search(r'(?:as |named |called )"?([A-Za-z0-9_\s]+)"?', query)
                name = name_match.group(1).strip() if name_match else None

                return {
                    'action': 'parse_molecule',
                    'structure': structure,
                    'format': format_type,
                    'name': name
                }

        # Calculate descriptors
        elif 'descriptor' in query_lower or 'property' in query_lower or 'properties' in query_lower:
            mol_match = re.search(r'mol_(\d+)', query)
            mol_id = f"mol_{mol_match.group(1)}" if mol_match else None

            descriptor_set = 'basic'
            if 'all' in query_lower:
                descriptor_set = 'all'
            elif 'lipinski' in query_lower:
                descriptor_set = 'lipinski'

            return {
                'action': 'descriptors',
                'mol_id': mol_id,
                'descriptor_set': descriptor_set
            }

        # Drug-likeness
        elif 'drug' in query_lower or 'likeness' in query_lower:
            mol_match = re.search(r'mol_(\d+)', query)
            mol_id = f"mol_{mol_match.group(1)}" if mol_match else None

            return {
                'action': 'drug_likeness',
                'mol_id': mol_id
            }

        # Generate fingerprint
        elif 'fingerprint' in query_lower:
            mol_match = re.search(r'mol_(\d+)', query)
            mol_id = f"mol_{mol_match.group(1)}" if mol_match else None

            fp_type = 'morgan'
            if 'maccs' in query_lower:
                fp_type = 'maccs'
            elif 'rdk' in query_lower:
                fp_type = 'rdk'
            elif 'atompair' in query_lower:
                fp_type = 'atompair'

            return {
                'action': 'fingerprint',
                'mol_id': mol_id,
                'fp_type': fp_type
            }

        # Similarity search
        elif 'similar' in query_lower or 'similarity' in query_lower:
            mol_match = re.search(r'mol_(\d+)', query)
            mol_id = f"mol_{mol_match.group(1)}" if mol_match else None

            return {
                'action': 'similarity',
                'query_mol_id': mol_id
            }

        # Substructure search
        elif 'substructure' in query_lower:
            pattern_match = re.search(r'(?:pattern |smarts )([A-Za-z0-9()=\[\]@#+\-]+)', query)
            if not pattern_match:
                pattern_match = re.search(r'"([^"]+)"', query)

            pattern = pattern_match.group(1) if pattern_match else None

            return {
                'action': 'substructure',
                'pattern': pattern
            }

        # List molecules
        elif 'list' in query_lower or 'show' in query_lower:
            return {'action': 'list_molecules'}

        # Status
        elif 'status' in query_lower or 'info' in query_lower:
            return {'action': 'status'}

        else:
            return {'action': 'help'}

    def execute(self, query: str):
        """Execute natural language command"""
        parsed = self.parse_command(query)
        action = parsed.get('action')

        if action == 'parse_molecule':
            result = self.agent.parse_molecule(
                structure=parsed['structure'],
                format=parsed['format'],
                name=parsed.get('name')
            )

            if result.get('success'):
                print(f"✅ Molecule parsed successfully!")
                print(f"   ID: {result['molecule_id']}")
                print(f"   Name: {result['name']}")
                print(f"   SMILES: {result['smiles']}")
                print(f"\n   Properties:")
                for key, value in result['properties'].items():
                    if isinstance(value, float):
                        print(f"      {key}: {value:.2f}")
                    else:
                        print(f"      {key}: {value}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'descriptors':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID (e.g., mol_1)")
                return

            result = self.agent.calculate_descriptors(
                mol_id=parsed['mol_id'],
                descriptor_set=parsed['descriptor_set']
            )

            if result.get('success'):
                print(f"✅ Descriptors calculated!")
                print(f"   Molecule: {parsed['mol_id']}")
                print(f"   Set: {result['descriptor_set']}")
                print(f"   Count: {result['count']}")
                print(f"\n   Descriptors:")
                for key, value in list(result['descriptors'].items())[:20]:
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")
                if result['count'] > 20:
                    print(f"   ... and {result['count'] - 20} more")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'drug_likeness':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID (e.g., mol_1)")
                return

            result = self.agent.drug_likeness_analysis(mol_id=parsed['mol_id'])

            if result.get('success'):
                print(f"✅ Drug-likeness analysis complete!")
                print(f"   Molecule: {parsed['mol_id']}")
                print(f"\n   Lipinski's Rule of Five:")
                lipinski = result['lipinski']
                print(f"      MW: {lipinski['molecular_weight']:.2f} (≤500)")
                print(f"      LogP: {lipinski['logp']:.2f} (≤5)")
                print(f"      HBD: {lipinski['hbd']} (≤5)")
                print(f"      HBA: {lipinski['hba']} (≤10)")
                print(f"      Violations: {lipinski['violations']}")
                print(f"      Status: {'✅ PASS' if lipinski['passes'] else '❌ FAIL'}")

                print(f"\n   Veber's Rules:")
                veber = result['veber']
                print(f"      Rotatable bonds: {veber['rotatable_bonds']} (≤10)")
                print(f"      TPSA: {veber['tpsa']:.2f} (≤140)")
                print(f"      Violations: {veber['violations']}")
                print(f"      Status: {'✅ PASS' if veber['passes'] else '❌ FAIL'}")

                if result.get('qed'):
                    print(f"\n   QED Score: {result['qed']:.3f} (0-1, higher is better)")

                print(f"\n   Overall: {result['overall_drug_likeness']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'fingerprint':
            if not parsed.get('mol_id'):
                print("❌ Please specify a molecule ID (e.g., mol_1)")
                return

            result = self.agent.generate_fingerprint(
                mol_id=parsed['mol_id'],
                fp_type=parsed['fp_type']
            )

            if result.get('success'):
                print(f"✅ Fingerprint generated!")
                print(f"   Molecule: {parsed['mol_id']}")
                print(f"   Type: {result['fp_type']}")
                print(f"   Bits: {result['n_bits']}")
                print(f"   On bits: {result['on_bits']}")
                if result.get('radius'):
                    print(f"   Radius: {result['radius']}")
                print(f"   Saved to: {result['fingerprint_file']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'similarity':
            if not parsed.get('query_mol_id'):
                print("❌ Please specify a query molecule ID (e.g., mol_1)")
                return

            result = self.agent.similarity_search(
                query_mol_id=parsed['query_mol_id']
            )

            if result.get('success'):
                print(f"✅ Similarity search complete!")
                print(f"   Query: {parsed['query_mol_id']}")
                print(f"   Metric: {result['metric']}")
                print(f"   Found: {result['count']} molecules")

                if result['results']:
                    print(f"\n   Top matches:")
                    for i, match in enumerate(result['results'][:10], 1):
                        print(f"   {i}. {match['molecule_id']} ({match['name']}): {match['similarity']:.4f}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'substructure':
            if not parsed.get('pattern'):
                print("❌ Please specify a substructure pattern")
                return

            result = self.agent.substructure_search(
                pattern=parsed['pattern'],
                pattern_format='smarts'
            )

            if result.get('success'):
                print(f"✅ Substructure search complete!")
                print(f"   Pattern: {result['pattern']}")
                print(f"   Matches: {result['count']}")

                if result['matches']:
                    print(f"\n   Found in:")
                    for match in result['matches']:
                        print(f"      {match['molecule_id']} ({match['name']})")
                        print(f"      SMILES: {match['smiles']}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'list_molecules':
            result = self.agent.list_molecules()

            if result.get('success'):
                print(f"🧪 Loaded molecules: {result['count']}")
                for mol in result['molecules']:
                    print(f"\n   [{mol['id']}] {mol['name']}")
                    print(f"      SMILES: {mol['smiles']}")
                    print(f"      MW: {mol['properties']['molecular_weight']:.2f}")
                    print(f"      LogP: {mol['properties']['logp']:.2f}")
            else:
                print(f"❌ Error: {result.get('error')}")

        elif action == 'status':
            status = self.agent.get_status()
            print(f"🧪 RDKit Agent Status")
            print(f"   Available: {status['available']}")
            print(f"   RDKit installed: {status['rdkit_installed']}")
            print(f"   Molecules: {status['molecules_loaded']}")
            print(f"   Storage: {status['storage_path']}")

        else:
            self.show_help()

    def show_help(self):
        """Show usage help"""
        print("""
🧪 RDKit Cheminformatics CLI - Natural Language Interface

Molecular Analysis, Drug Discovery, Chemical Fingerprinting

USAGE:
    python3 rdkit_cli.py "your natural language command"

EXAMPLES:
    # Parse molecules
    python3 rdkit_cli.py "parse molecule CCO as Ethanol"
    python3 rdkit_cli.py "analyze molecule CC(=O)Oc1ccccc1C(=O)O as Aspirin"

    # Calculate descriptors
    python3 rdkit_cli.py "calculate descriptors for mol_1"
    python3 rdkit_cli.py "calculate all descriptors for mol_1"

    # Drug-likeness
    python3 rdkit_cli.py "analyze drug likeness for mol_1"

    # Fingerprints
    python3 rdkit_cli.py "generate morgan fingerprint for mol_1"
    python3 rdkit_cli.py "generate maccs fingerprint for mol_1"

    # Similarity search
    python3 rdkit_cli.py "find molecules similar to mol_1"

    # Substructure search
    python3 rdkit_cli.py "search for substructure c1ccccc1"

    # Management
    python3 rdkit_cli.py "list molecules"
    python3 rdkit_cli.py "status"

DEPENDENCIES:
    pip install rdkit pandas numpy

TEMPEST COMPLIANCE:
    - All processing local (air-gapped compatible)
    - No cloud dependencies
    - Suitable for classified drug discovery research
    - Standard EM emissions profile
        """)

def main():
    if len(sys.argv) < 2:
        cli = RDKitCLI()
        cli.show_help()
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    cli = RDKitCLI()
    cli.execute(query)

if __name__ == "__main__":
    main()
