#!/usr/bin/env python3
"""
Heretic Abliteration Engine - Core Model Manipulation

Implements directional ablation (abliteration) for removing safety constraints
from transformer-based language models through orthogonal projection.

Based on: https://github.com/p-e-w/heretic
Paper: "Refusal in Language Models Is Mediated by a Single Direction" (Arditi et al. 2024)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class AbliterationParameters:
    """
    Parameters for abliteration at a specific layer component.

    Attributes:
        max_weight: Peak ablation magnitude (0.0 to 5.0)
        max_weight_position: Layer index of maximum effect (0.0 to 1.0, fraction of total layers)
        min_weight: Baseline ablation weight (0.0 to max_weight)
        min_weight_distance: Transition width in layers (0.0 to n_layers/2)
    """
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "max_weight": self.max_weight,
            "max_weight_position": self.max_weight_position,
            "min_weight": self.min_weight,
            "min_weight_distance": self.min_weight_distance
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AbliterationParameters':
        """Load from dictionary"""
        return cls(
            max_weight=data["max_weight"],
            max_weight_position=data["max_weight_position"],
            min_weight=data["min_weight"],
            min_weight_distance=data["min_weight_distance"]
        )


class RefusalDirectionCalculator:
    """
    Calculate refusal directions from harmless and harmful prompt datasets.

    Refusal directions are computed as the difference-of-means between residual
    activations from harmful prompts vs harmless prompts, normalized per layer.

    Formula: r_l = normalize(mean(h_bad,l) - mean(h_good,l))
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize refusal direction calculator.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            device: Device for computation (cuda, cpu, xpu, etc.)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def get_layers(self):
        """
        Extract transformer layers with multimodal fallback.

        Returns:
            ModuleList of transformer layers
        """
        try:
            # Multimodal models (LLaVA, Qwen-VL, etc.)
            return self.model.model.text_model.layers
        except AttributeError:
            # Text-only models (Llama, Qwen, Mistral, etc.)
            return self.model.model.layers

    def get_residuals(self, prompts: List[str], batch_size: int = 8) -> torch.Tensor:
        """
        Extract residual activations at final token position for all layers.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for processing

        Returns:
            Tensor of shape [n_prompts, n_layers, hidden_size]
        """
        all_residuals = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                # Run forward pass with output_hidden_states
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]

                # Extract final token position for each layer
                batch_residuals = []
                for layer_hidden in hidden_states[:-1]:  # Exclude final layer
                    # Get last token position (before padding)
                    last_token_idx = inputs.attention_mask.sum(dim=1) - 1
                    layer_residual = torch.stack([
                        layer_hidden[j, last_token_idx[j], :]
                        for j in range(layer_hidden.shape[0])
                    ])
                    batch_residuals.append(layer_residual)

                # Stack layers: [batch, n_layers, hidden_size]
                batch_residuals = torch.stack(batch_residuals, dim=1)
                all_residuals.append(batch_residuals.cpu().float())

        # Concatenate all batches: [n_prompts, n_layers, hidden_size]
        return torch.cat(all_residuals, dim=0)

    def calculate_refusal_directions(
        self,
        good_prompts: List[str],
        bad_prompts: List[str],
        batch_size: int = 8
    ) -> torch.Tensor:
        """
        Calculate refusal directions from harmless and harmful prompts.

        Algorithm:
        1. Extract residuals for good and bad prompts
        2. Compute mean residuals per layer
        3. Calculate difference: bad_mean - good_mean
        4. Normalize with L2 norm per layer

        Args:
            good_prompts: List of harmless prompts
            bad_prompts: List of harmful prompts
            batch_size: Batch size for processing

        Returns:
            Tensor of shape [n_layers, hidden_size] with refusal directions
        """
        logger.info(f"Extracting residuals from {len(good_prompts)} good prompts...")
        good_residuals = self.get_residuals(good_prompts, batch_size)

        logger.info(f"Extracting residuals from {len(bad_prompts)} bad prompts...")
        bad_residuals = self.get_residuals(bad_prompts, batch_size)

        # Compute means: [n_layers, hidden_size]
        good_mean = good_residuals.mean(dim=0)
        bad_mean = bad_residuals.mean(dim=0)

        # Difference of means
        refusal_directions = bad_mean - good_mean

        # L2 normalize per layer
        refusal_directions = F.normalize(refusal_directions, p=2, dim=-1)

        logger.info(f"Calculated refusal directions: shape {refusal_directions.shape}")
        return refusal_directions

    def save_refusal_directions(
        self,
        refusal_directions: torch.Tensor,
        save_path: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save refusal directions to disk with metadata.

        Args:
            refusal_directions: Tensor of refusal directions
            save_path: Path to save file
            metadata: Optional metadata dictionary
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensor
        torch.save(refusal_directions, save_path)

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update({
            "shape": list(refusal_directions.shape),
            "dtype": str(refusal_directions.dtype),
            "hash": hashlib.sha256(refusal_directions.numpy().tobytes()).hexdigest()
        })

        meta_path = save_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved refusal directions to {save_path}")

    @staticmethod
    def load_refusal_directions(load_path: Path) -> Tuple[torch.Tensor, Dict]:
        """
        Load refusal directions from disk.

        Args:
            load_path: Path to load file

        Returns:
            Tuple of (refusal_directions tensor, metadata dict)
        """
        load_path = Path(load_path)

        # Load tensor
        refusal_directions = torch.load(load_path)

        # Load metadata
        meta_path = load_path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return refusal_directions, metadata


class ModelAbliterator:
    """
    Apply abliteration to transformer models using orthogonal projection.

    Core Algorithm:
        W_new = W_old - α * (r ⊗ r) @ W_old

    Where:
        - W = weight matrix
        - α = layer-specific ablation weight
        - r = normalized refusal direction
        - ⊗ = outer product
    """

    def __init__(self, model):
        """
        Initialize model abliterator.

        Args:
            model: Loaded transformer model
        """
        self.model = model
        self.layers = self._get_layers()

    def _get_layers(self):
        """Extract transformer layers"""
        try:
            return self.model.model.text_model.layers
        except AttributeError:
            return self.model.model.layers

    def get_layer_matrices(self, layer_index: int) -> Dict[str, List[torch.Tensor]]:
        """
        Extract weight matrices from specific layer components.

        Targets:
            - Attention: attn.o_proj
            - MLP: mlp.down_proj (dense) or mlp.experts[i].down_proj (MoE)

        Args:
            layer_index: Index of layer to extract

        Returns:
            Dict mapping component names to lists of weight tensors
        """
        layer = self.layers[layer_index]
        matrices = {"attn": [], "mlp": []}

        # Attention output projection
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
            matrices["attn"].append(layer.self_attn.o_proj.weight)
        elif hasattr(layer, 'attn') and hasattr(layer.attn, 'o_proj'):
            matrices["attn"].append(layer.attn.o_proj.weight)

        # MLP down-projection
        if hasattr(layer, 'mlp'):
            mlp = layer.mlp

            # Dense models
            if hasattr(mlp, 'down_proj'):
                matrices["mlp"].append(mlp.down_proj.weight)

            # MoE models
            if hasattr(mlp, 'shared_expert') and hasattr(mlp.shared_expert, 'down_proj'):
                matrices["mlp"].append(mlp.shared_expert.down_proj.weight)

            if hasattr(mlp, 'experts'):
                for expert in mlp.experts:
                    if hasattr(expert, 'down_proj'):
                        matrices["mlp"].append(expert.down_proj.weight)

        return matrices

    def interpolate_direction(
        self,
        refusal_directions: torch.Tensor,
        direction_index: Optional[float]
    ) -> torch.Tensor:
        """
        Interpolate refusal direction using fractional layer indexing.

        If direction_index is None, returns mean across all layers (global).
        If fractional (e.g., 0.75), interpolates between floor and ceiling layers.

        Args:
            refusal_directions: Tensor of shape [n_layers, hidden_size]
            direction_index: Layer index (can be fractional) or None for global

        Returns:
            Single refusal direction vector of shape [hidden_size]
        """
        if direction_index is None:
            # Global direction (mean across layers)
            return refusal_directions.mean(dim=0)

        # Fractional interpolation
        floor_idx = int(direction_index)
        ceil_idx = min(floor_idx + 1, refusal_directions.shape[0] - 1)
        weight = direction_index - floor_idx

        refusal_direction = (
            (1 - weight) * refusal_directions[floor_idx] +
            weight * refusal_directions[ceil_idx]
        )

        return refusal_direction

    def calculate_layer_weight(
        self,
        layer_idx: int,
        n_layers: int,
        params: AbliterationParameters
    ) -> float:
        """
        Calculate ablation weight for a layer using distance-based kernel.

        Algorithm:
            distance = |layer_idx - max_weight_position|
            if distance > min_weight_distance:
                return 0.0  # Skip distant layers
            else:
                return linear_interpolation(max_weight, min_weight, distance)

        Args:
            layer_idx: Current layer index
            n_layers: Total number of layers
            params: Abliteration parameters

        Returns:
            Ablation weight for this layer (0.0 if outside kernel)
        """
        # Convert fractional position to absolute index
        max_pos_abs = params.max_weight_position * (n_layers - 1)
        distance = abs(layer_idx - max_pos_abs)

        if distance > params.min_weight_distance:
            return 0.0  # Outside kernel, no ablation

        # Linear interpolation from max_weight to min_weight
        weight_range = params.max_weight - params.min_weight
        layer_weight = params.max_weight - (weight_range * distance / params.min_weight_distance)

        return layer_weight

    def abliterate(
        self,
        refusal_directions: torch.Tensor,
        direction_index: Optional[float],
        parameters: Dict[str, AbliterationParameters]
    ):
        """
        Apply abliteration to model using orthogonal projection.

        For each layer and component:
        1. Interpolate refusal direction (if fractional index)
        2. Calculate layer weight using distance kernel
        3. Create projection matrix: P = r ⊗ r
        4. Update weights in-place: W -= α * P @ W

        Args:
            refusal_directions: Tensor of shape [n_layers, hidden_size]
            direction_index: Layer index (fractional) or None for global
            parameters: Dict mapping component names to AbliterationParameters
        """
        n_layers = len(self.layers)

        # Interpolate refusal direction (global or layer-specific)
        refusal_direction = self.interpolate_direction(refusal_directions, direction_index)
        refusal_direction = F.normalize(refusal_direction, p=2, dim=-1)

        # Move to model device
        device = next(self.model.parameters()).device
        refusal_direction = refusal_direction.to(device)

        # Create projection matrix: P = r ⊗ r
        projector = torch.outer(refusal_direction, refusal_direction)

        logger.info(f"Applying abliteration to {n_layers} layers...")

        for layer_idx in range(n_layers):
            # Get weight matrices for this layer
            layer_matrices = self.get_layer_matrices(layer_idx)

            # Apply abliteration to each component
            for component_name, matrices in layer_matrices.items():
                if component_name not in parameters:
                    continue

                params = parameters[component_name]

                # Calculate layer weight using distance kernel
                layer_weight = self.calculate_layer_weight(layer_idx, n_layers, params)

                if layer_weight == 0.0:
                    continue  # Skip this layer

                # Apply orthogonal projection to all matrices in component
                for matrix in matrices:
                    # W_new = W_old - α * P @ W_old
                    matrix.sub_(layer_weight * (projector @ matrix))

        logger.info("Abliteration complete!")


class HereticModelWrapper:
    """
    High-level wrapper for abliteration workflows.

    Combines RefusalDirectionCalculator and ModelAbliterator for easy use.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize wrapper.

        Args:
            model: Loaded transformer model
            tokenizer: Corresponding tokenizer
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        self.direction_calculator = RefusalDirectionCalculator(model, tokenizer, device)
        self.abliterator = ModelAbliterator(model)

    def full_abliteration_workflow(
        self,
        good_prompts: List[str],
        bad_prompts: List[str],
        parameters: Dict[str, AbliterationParameters],
        direction_index: Optional[float] = None,
        save_directions_path: Optional[Path] = None
    ) -> torch.Tensor:
        """
        Complete abliteration workflow from prompts to abliterated model.

        Steps:
        1. Calculate refusal directions
        2. Optionally save directions
        3. Apply abliteration
        4. Return refusal directions for evaluation

        Args:
            good_prompts: Harmless prompts
            bad_prompts: Harmful prompts
            parameters: Abliteration parameters per component
            direction_index: Fractional layer index or None
            save_directions_path: Optional path to save refusal directions

        Returns:
            Refusal directions tensor
        """
        # Calculate refusal directions
        refusal_directions = self.direction_calculator.calculate_refusal_directions(
            good_prompts, bad_prompts
        )

        # Optionally save
        if save_directions_path:
            self.direction_calculator.save_refusal_directions(
                refusal_directions,
                save_directions_path,
                metadata={
                    "n_good_prompts": len(good_prompts),
                    "n_bad_prompts": len(bad_prompts),
                    "direction_index": direction_index,
                    "parameters": {k: v.to_dict() for k, v in parameters.items()}
                }
            )

        # Apply abliteration
        self.abliterator.abliterate(refusal_directions, direction_index, parameters)

        return refusal_directions


# Utility functions

def empty_cache():
    """Clear GPU memory across all accelerator types"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if hasattr(torch, 'mlu') and torch.mlu.is_available():
        torch.mlu.empty_cache()


def detect_accelerator() -> str:
    """Detect available accelerator"""
    if torch.cuda.is_available():
        return "CUDA"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "XPU"
    elif hasattr(torch, 'mlu') and torch.mlu.is_available():
        return "MLU"
    else:
        return "CPU"


if __name__ == "__main__":
    # Example usage
    print("Heretic Abliteration Engine")
    print("=" * 60)
    print(f"Accelerator: {detect_accelerator()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
