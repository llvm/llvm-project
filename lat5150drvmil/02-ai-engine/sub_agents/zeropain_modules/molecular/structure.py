#!/usr/bin/env python3
"""
Molecular Structure Module
SMILES processing, 3D structure generation, and molecular representations
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


@dataclass
class MolecularStructure:
    """Complete molecular structure representation"""
    name: str
    smiles: str
    mol_weight: float
    logp: float
    tpsa: float  # Topological Polar Surface Area
    h_donors: int
    h_acceptors: int
    rotatable_bonds: int
    aromatic_rings: int
    formal_charge: int
    mol_formula: str

    # Drug-likeness
    lipinski_violations: int
    bioavailability_score: float

    # Optional 3D data
    coordinates_3d: Optional[np.ndarray] = None
    conformer_energy: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'smiles': self.smiles,
            'molecular_weight': self.mol_weight,
            'logP': self.logp,
            'tpsa': self.tpsa,
            'h_donors': self.h_donors,
            'h_acceptors': self.h_acceptors,
            'rotatable_bonds': self.rotatable_bonds,
            'aromatic_rings': self.aromatic_rings,
            'formal_charge': self.formal_charge,
            'molecular_formula': self.mol_formula,
            'lipinski_violations': self.lipinski_violations,
            'bioavailability_score': self.bioavailability_score,
            'has_3d': self.coordinates_3d is not None
        }

    def passes_lipinski(self) -> bool:
        """Check if passes Lipinski's Rule of Five"""
        return self.lipinski_violations == 0

    def is_drug_like(self) -> bool:
        """Enhanced drug-likeness check"""
        return (
            self.passes_lipinski() and
            self.bioavailability_score > 0.5 and
            self.rotatable_bonds <= 10 and
            self.formal_charge in [-1, 0, 1]
        )


def from_smiles(smiles: str, name: str = "Compound",
                generate_3d: bool = True) -> Optional[MolecularStructure]:
    """
    Create MolecularStructure from SMILES string

    Args:
        smiles: SMILES string
        name: Compound name
        generate_3d: Generate 3D coordinates

    Returns:
        MolecularStructure or None if invalid SMILES
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required. Install with: conda install -c conda-forge rdkit")

    try:
        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Add hydrogens for accurate property calculation
        mol = Chem.AddHs(mol)

        # Calculate 2D descriptors
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        aromatic = Descriptors.NumAromaticRings(mol)
        charge = Chem.GetFormalCharge(mol)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

        # Lipinski's Rule of Five
        lipinski_violations = 0
        if mw > 500:
            lipinski_violations += 1
        if logp > 5:
            lipinski_violations += 1
        if hbd > 5:
            lipinski_violations += 1
        if hba > 10:
            lipinski_violations += 1

        # Bioavailability score (simplified)
        bioavail = 1.0
        if rotatable > 10:
            bioavail *= 0.7
        if tpsa > 140:
            bioavail *= 0.8
        if mw > 500:
            bioavail *= 0.6
        bioavail = np.clip(bioavail, 0.0, 1.0)

        # Generate 3D coordinates if requested
        coords_3d = None
        conformer_energy = None

        if generate_3d:
            try:
                # Embed molecule
                AllChem.EmbedMolecule(mol, randomSeed=42)
                # Optimize geometry
                result = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)

                if result == 0:  # Success
                    # Get 3D coordinates
                    conf = mol.GetConformer()
                    coords_3d = np.array([
                        [conf.GetAtomPosition(i).x,
                         conf.GetAtomPosition(i).y,
                         conf.GetAtomPosition(i).z]
                        for i in range(mol.GetNumAtoms())
                    ])

                    # Calculate conformer energy
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    if props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, props)
                        conformer_energy = ff.CalcEnergy()

            except Exception as e:
                print(f"3D generation failed: {e}")

        return MolecularStructure(
            name=name,
            smiles=smiles,
            mol_weight=mw,
            logp=logp,
            tpsa=tpsa,
            h_donors=hbd,
            h_acceptors=hba,
            rotatable_bonds=rotatable,
            aromatic_rings=aromatic,
            formal_charge=charge,
            mol_formula=formula,
            lipinski_violations=lipinski_violations,
            bioavailability_score=bioavail,
            coordinates_3d=coords_3d,
            conformer_energy=conformer_energy
        )

    except Exception as e:
        print(f"Error processing SMILES: {e}")
        return None


def smiles_to_mol2(smiles: str, output_file: str) -> bool:
    """Convert SMILES to MOL2 format"""
    if not RDKIT_AVAILABLE:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Write MOL2
        writer = Chem.SDWriter(output_file.replace('.mol2', '.sdf'))
        writer.write(mol)
        writer.close()

        return True
    except Exception as e:
        print(f"MOL2 conversion error: {e}")
        return False


def generate_2d_image(smiles: str, output_file: str, size=(300, 300)) -> bool:
    """Generate 2D structure image"""
    if not RDKIT_AVAILABLE:
        return False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        img = Draw.MolToImage(mol, size=size)
        img.save(output_file)
        return True
    except Exception as e:
        print(f"Image generation error: {e}")
        return False


def calculate_similarity(smiles1: str, smiles2: str) -> float:
    """
    Calculate Tanimoto similarity between two molecules

    Args:
        smiles1: First SMILES string
        smiles2: Second SMILES string

    Returns:
        Similarity score (0-1)
    """
    if not RDKIT_AVAILABLE:
        return 0.0

    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return 0.0

        # Morgan fingerprints
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)

        # Tanimoto similarity
        from rdkit import DataStructs
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

        return similarity

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0


if __name__ == '__main__':
    print("ZeroPain Molecular Structure Module")
    print("=" * 60)

    # Test with morphine
    morphine_smiles = "CN1CC[C@]23[C@H]4Oc5c(O)ccc(C[C@@H]1[C@@H]2C=C[C@@H]4O)c35"

    print(f"\nAnalyzing morphine...")
    print(f"SMILES: {morphine_smiles}\n")

    struct = from_smiles(morphine_smiles, "Morphine", generate_3d=True)

    if struct:
        print("Molecular Properties:")
        print(f"  Formula:          {struct.mol_formula}")
        print(f"  Molecular Weight: {struct.mol_weight:.2f} g/mol")
        print(f"  LogP:             {struct.logp:.2f}")
        print(f"  TPSA:             {struct.tpsa:.2f} Ų")
        print(f"  H Donors:         {struct.h_donors}")
        print(f"  H Acceptors:      {struct.h_acceptors}")
        print(f"  Rotatable Bonds:  {struct.rotatable_bonds}")
        print(f"  Aromatic Rings:   {struct.aromatic_rings}")
        print(f"\nDrug-likeness:")
        print(f"  Lipinski Violations: {struct.lipinski_violations}")
        print(f"  Bioavailability:     {struct.bioavailability_score*100:.1f}%")
        print(f"  Drug-like:           {'Yes' if struct.is_drug_like() else 'No'}")

        if struct.coordinates_3d is not None:
            print(f"\n3D Structure:")
            print(f"  Atoms:            {len(struct.coordinates_3d)}")
            print(f"  Conformer Energy: {struct.conformer_energy:.2f} kcal/mol" if struct.conformer_energy else "  Energy: N/A")
