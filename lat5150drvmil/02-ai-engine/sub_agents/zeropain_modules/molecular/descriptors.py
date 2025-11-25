#!/usr/bin/env python3
"""
Molecular Descriptors Module
Calculate comprehensive molecular descriptors for QSAR and ML models
"""

from typing import Dict, List
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors
    from rdkit.Chem import rdMolDescriptors, rdPartialCharges
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def calculate_descriptors(smiles: str) -> Dict:
    """
    Calculate comprehensive molecular descriptors

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of descriptors
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)

    descriptors = {
        # Basic properties
        'molecular_weight': Descriptors.MolWt(mol),
        'exact_mass': Descriptors.ExactMolWt(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),

        # Lipophilicity
        'logP': Crippen.MolLogP(mol),
        'MR': Crippen.MolMR(mol),  # Molar refractivity

        # Topology
        'tpsa': Descriptors.TPSA(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_heavy_atoms': Lipinski.HeavyAtomCount(mol),
        'num_heteroatoms': Lipinski.NumHeteroatoms(mol),

        # Hydrogen bonding
        'h_donors': Descriptors.NumHDonors(mol),
        'h_acceptors': Descriptors.NumHAcceptors(mol),

        # Flexibility
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'fraction_sp3': Descriptors.FractionCsp3(mol),

        # Rings
        'num_rings': Descriptors.RingCount(mol),
        'aromatic_rings': Descriptors.NumAromaticRings(mol),
        'aliphatic_rings': Descriptors.NumAliphaticRings(mol),

        # Charge
        'formal_charge': Chem.GetFormalCharge(mol),

        # Complexity
        'bertz_ct': GraphDescriptors.BertzCT(mol),  # Complexity

        # Drug-likeness
        'qed': Descriptors.qed(mol),  # Quantitative Estimate of Drug-likeness

        # Synthetic accessibility
        'num_stereocenters': Descriptors.NumSaturatedCarbocycles(mol),
    }

    # Lipinski's Rule of Five
    descriptors['lipinski_violations'] = sum([
        descriptors['molecular_weight'] > 500,
        descriptors['logP'] > 5,
        descriptors['h_donors'] > 5,
        descriptors['h_acceptors'] > 10
    ])

    # Veber's Rule (for oral bioavailability)
    descriptors['veber_violations'] = sum([
        descriptors['rotatable_bonds'] > 10,
        descriptors['tpsa'] > 140
    ])

    # Ghose filter (for drug-likeness)
    descriptors['ghose_violations'] = sum([
        descriptors['molecular_weight'] < 160 or descriptors['molecular_weight'] > 480,
        descriptors['logP'] < -0.4 or descriptors['logP'] > 5.6,
        descriptors['num_atoms'] < 20 or descriptors['num_atoms'] > 70,
        descriptors['MR'] < 40 or descriptors['MR'] > 130
    ])

    return descriptors


def calculate_fingerprint(smiles: str, fp_type: str = 'morgan') -> np.ndarray:
    """
    Calculate molecular fingerprint

    Args:
        smiles: SMILES string
        fp_type: Fingerprint type (morgan, maccs, topological)

    Returns:
        Numpy array of fingerprint bits
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    from rdkit.Chem import AllChem, MACCSkeys

    if fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    elif fp_type == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == 'topological':
        fp = Chem.RDKFingerprint(mol)
    else:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")

    return np.array(fp)


if __name__ == '__main__':
    morphine = "CN1CC[C@]23[C@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]4O)c35"

    descriptors = calculate_descriptors(morphine)

    print("Molecular Descriptors for Morphine:")
    print("=" * 60)
    for key, value in descriptors.items():
        print(f"  {key:25s}: {value}")
