#!/usr/bin/env python3
"""
Binding Analysis Module
Analyze protein-ligand interactions and binding mechanisms
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class InteractionProfile:
    """Detailed interaction profile"""
    hydrogen_bonds: List[Dict]
    hydrophobic_contacts: List[Dict]
    pi_stacking: List[Dict]
    salt_bridges: List[Dict]
    water_bridges: List[Dict]
    binding_energy_breakdown: Dict[str, float]

    def total_interactions(self) -> int:
        return (
            len(self.hydrogen_bonds) +
            len(self.hydrophobic_contacts) +
            len(self.pi_stacking) +
            len(self.salt_bridges) +
            len(self.water_bridges)
        )


class BindingAnalyzer:
    """Analyze protein-ligand binding interactions"""

    def __init__(self):
        # Key residues in opioid receptors
        self.mor_binding_site = [
            'Asp147', 'Tyr148', 'Met151', 'Trp293', 'His297',
            'Ile296', 'Val236', 'Lys233', 'Tyr326', 'Ile322'
        ]

        self.dor_binding_site = [
            'Asp128', 'Tyr129', 'Met132', 'Trp274', 'His278',
            'Ile277', 'Val217', 'Lys214', 'Tyr308', 'Ile304'
        ]

        self.kor_binding_site = [
            'Asp138', 'Tyr139', 'Met142', 'Trp287', 'His291',
            'Ile290', 'Val230', 'Lys227', 'Tyr320', 'Ile316'
        ]

    def analyze_binding_pose(self, pdbqt_content: str,
                            receptor: str = "MOR") -> InteractionProfile:
        """
        Analyze binding pose for interactions

        Args:
            pdbqt_content: PDBQT file content
            receptor: Receptor type

        Returns:
            InteractionProfile
        """
        # In production, would use ProDy, MDAnalysis, or similar
        # For now, returning example profile

        h_bonds = [
            {
                'donor': 'N1',
                'acceptor': 'Asp147:OD1',
                'distance': 2.8,
                'angle': 165.0,
                'energy': -3.2
            },
            {
                'donor': 'O2',
                'acceptor': 'Tyr148:OH',
                'distance': 2.9,
                'angle': 158.0,
                'energy': -2.8
            }
        ]

        hydrophobic = [
            {
                'ligand_atom': 'C12',
                'residue': 'Val236',
                'distance': 3.8,
                'energy': -1.2
            },
            {
                'ligand_atom': 'C8',
                'residue': 'Ile296',
                'distance': 3.5,
                'energy': -1.5
            }
        ]

        pi_stack = [
            {
                'ligand_ring': 'Ring1',
                'residue': 'Trp293',
                'distance': 3.7,
                'angle': 12.0,
                'energy': -2.5
            }
        ]

        salt_bridges = []  # No charged interactions in this example

        water_bridges = [
            {
                'ligand_atom': 'O3',
                'water_id': 'WAT501',
                'protein_atom': 'His297:ND1',
                'energy': -1.8
            }
        ]

        energy_breakdown = {
            'hydrogen_bonding': -6.0,
            'hydrophobic': -2.7,
            'pi_stacking': -2.5,
            'electrostatic': 0.0,
            'water_mediated': -1.8,
            'desolvation': 2.5,
            'entropic': 3.0,
            'total': -7.5
        }

        return InteractionProfile(
            hydrogen_bonds=h_bonds,
            hydrophobic_contacts=hydrophobic,
            pi_stacking=pi_stack,
            salt_bridges=salt_bridges,
            water_bridges=water_bridges,
            binding_energy_breakdown=energy_breakdown
        )

    def calculate_selectivity(self, ki_mor: float, ki_dor: float,
                             ki_kor: float) -> Dict[str, float]:
        """
        Calculate receptor selectivity

        Args:
            ki_mor: MOR Ki (nM)
            ki_dor: DOR Ki (nM)
            ki_kor: KOR Ki (nM)

        Returns:
            Selectivity ratios
        """
        return {
            'MOR_vs_DOR': ki_dor / ki_mor if ki_mor > 0 else 1.0,
            'MOR_vs_KOR': ki_kor / ki_mor if ki_mor > 0 else 1.0,
            'DOR_vs_KOR': ki_kor / ki_dor if ki_dor > 0 else 1.0,
            'primary_target': self._determine_primary(ki_mor, ki_dor, ki_kor)
        }

    def _determine_primary(self, ki_mor: float, ki_dor: float,
                          ki_kor: float) -> str:
        """Determine primary receptor target"""
        min_ki = min(ki_mor, ki_dor, ki_kor)

        if min_ki == ki_mor:
            return "MOR (μ-opioid)"
        elif min_ki == ki_dor:
            return "DOR (δ-opioid)"
        else:
            return "KOR (κ-opioid)"

    def predict_signaling_bias(self, molecular_features: Dict) -> Dict[str, float]:
        """
        Predict G-protein vs β-arrestin bias from molecular features

        Args:
            molecular_features: Dictionary of molecular properties

        Returns:
            Predicted bias factors
        """
        # Simplified model - in reality would use trained ML model
        logp = molecular_features.get('logP', 0.0)
        aromatic = molecular_features.get('aromatic_rings', 0)
        rigid = molecular_features.get('rotatable_bonds', 0)

        # More rigid, aromatic compounds tend to be more biased
        g_protein_bias = 1.0 + 0.5 * aromatic - 0.1 * rigid
        beta_arrestin_bias = 1.0 - 0.3 * aromatic + 0.05 * rigid

        g_protein_bias = np.clip(g_protein_bias, 0.1, 15.0)
        beta_arrestin_bias = np.clip(beta_arrestin_bias, 0.01, 2.0)

        return {
            'g_protein_bias': g_protein_bias,
            'beta_arrestin_bias': beta_arrestin_bias,
            'bias_ratio': g_protein_bias / beta_arrestin_bias if beta_arrestin_bias > 0 else 10.0,
            'classification': self._classify_bias(g_protein_bias / beta_arrestin_bias)
        }

    def _classify_bias(self, ratio: float) -> str:
        """Classify signaling bias"""
        if ratio > 5:
            return "Highly G-protein biased (safer profile)"
        elif ratio > 2:
            return "Moderately G-protein biased"
        elif ratio > 0.5:
            return "Balanced signaling"
        else:
            return "β-arrestin biased (higher side effect risk)"


if __name__ == '__main__':
    analyzer = BindingAnalyzer()

    # Example analysis
    profile = analyzer.analyze_binding_pose("", "MOR")

    print("Binding Interaction Analysis")
    print("=" * 60)
    print(f"Total interactions: {profile.total_interactions()}")
    print(f"\nHydrogen bonds: {len(profile.hydrogen_bonds)}")
    for hb in profile.hydrogen_bonds:
        print(f"  {hb['donor']} -> {hb['acceptor']}: {hb['distance']:.2f}Å")

    print(f"\nHydrophobic contacts: {len(profile.hydrophobic_contacts)}")
    print(f"π-stacking: {len(profile.pi_stacking)}")

    print(f"\nEnergy breakdown:")
    for component, energy in profile.binding_energy_breakdown.items():
        print(f"  {component:20s}: {energy:6.2f} kcal/mol")

    # Selectivity example
    selectivity = analyzer.calculate_selectivity(1.8, 45.0, 120.0)
    print(f"\nSelectivity:")
    print(f"  MOR vs DOR: {selectivity['MOR_vs_DOR']:.1f}x")
    print(f"  Primary target: {selectivity['primary_target']}")
