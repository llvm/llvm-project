#!/usr/bin/env python3
"""
Heretic Web API - Flask Routes for DSMIL TEMPEST Interface

Provides REST API endpoints for model abliteration operations
to be integrated into the DSMIL TEMPEST web dashboard.
"""

from flask import Blueprint, request, jsonify
from pathlib import Path
import json
import threading
from typing import Dict, Any, Optional
from datetime import datetime

# Import Heretic modules
try:
    from heretic_config import ConfigLoader, HereticSettings
    from heretic_datasets import DatasetRegistry, CustomPromptBuilder
    from enhanced_ai_engine import EnhancedAIEngine, HERETIC_AVAILABLE
    from heretic_enhanced_abliteration import (
        EnhancedAbliterationConfig,
        AbliterationMethod,
        LLMJudge
    )
    from heretic_unsloth_integration import UnslothConfig, UnslothOptimizer
    HERETIC_WEB_AVAILABLE = True
    ENHANCED_HERETIC_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Heretic modules not available: {e}")
    HERETIC_WEB_AVAILABLE = False
    HERETIC_AVAILABLE = False
    ENHANCED_HERETIC_AVAILABLE = False

# Create Blueprint
heretic_bp = Blueprint('heretic', __name__, url_prefix='/api/heretic')

# Global state for async operations
abliteration_jobs = {}
job_counter = 0


# ===== CONFIGURATION ENDPOINTS =====

@heretic_bp.route('/config', methods=['GET'])
def get_config():
    """
    Get current Heretic configuration.

    Returns:
        JSON with configuration settings
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        config_path = Path(__file__).parent / "heretic_config.toml"
        settings = ConfigLoader.load(config_file=config_path if config_path.exists() else None)

        return jsonify({
            "optimization": {
                "n_trials": settings.n_trials,
                "n_startup_trials": settings.n_startup_trials,
                "max_batch_size": settings.max_batch_size,
                "kl_divergence_scale": settings.kl_divergence_scale
            },
            "datasets": {
                "good_prompts": settings.good_prompts_dataset,
                "bad_prompts": settings.bad_prompts_dataset,
                "good_eval": settings.good_eval_dataset,
                "bad_eval": settings.bad_eval_dataset
            },
            "storage": {
                "models": str(settings.abliterated_models_dir),
                "directions": str(settings.refusal_directions_dir),
                "results": str(settings.optimization_results_dir)
            },
            "refusal_detection": {
                "marker_count": len(settings.refusal_markers),
                "markers": settings.refusal_markers[:5] + ["..."]  # First 5 only
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/config', methods=['POST'])
def update_config():
    """
    Update Heretic configuration.

    Request body:
        {
            "n_trials": 200,
            "n_startup_trials": 60,
            "max_batch_size": 128,
            ...
        }

    Returns:
        JSON with updated configuration
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        data = request.json
        config_path = Path(__file__).parent / "heretic_config.toml"

        # Load current config
        settings = ConfigLoader.load(config_file=config_path if config_path.exists() else None)

        # Update with new values
        if "n_trials" in data:
            settings.n_trials = int(data["n_trials"])
        if "n_startup_trials" in data:
            settings.n_startup_trials = int(data["n_startup_trials"])
        if "max_batch_size" in data:
            settings.max_batch_size = int(data["max_batch_size"])
        if "kl_divergence_scale" in data:
            settings.kl_divergence_scale = float(data["kl_divergence_scale"])

        # Save updated config
        # (In production, would write back to TOML file)

        return jsonify({
            "status": "updated",
            "message": "Configuration updated successfully",
            "n_trials": settings.n_trials
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== DATASET ENDPOINTS =====

@heretic_bp.route('/datasets', methods=['GET'])
def list_datasets():
    """
    List available datasets.

    Returns:
        JSON with dataset information
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        registry = DatasetRegistry()

        datasets = [
            {
                "name": name,
                "type": "harmless" if "harmless" in name else "harmful",
                "split": "train" if "train" in name else "test"
            }
            for name in registry.list_datasets()
        ]

        # Add custom prompts
        code_prompts = CustomPromptBuilder.create_code_prompts()
        medical_prompts = CustomPromptBuilder.create_medical_prompts()

        custom_sets = [
            {
                "name": "code",
                "type": "custom",
                "good_count": len(code_prompts["good"]),
                "bad_count": len(code_prompts["bad"])
            },
            {
                "name": "medical",
                "type": "custom",
                "good_count": len(medical_prompts["good"]),
                "bad_count": len(medical_prompts["bad"])
            }
        ]

        return jsonify({
            "registered": datasets,
            "custom": custom_sets
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== MODEL ABLITERATION ENDPOINTS =====

@heretic_bp.route('/abliterate', methods=['POST'])
def abliterate_model():
    """
    Start model abliteration workflow (async).

    Request body:
        {
            "model": "uncensored_code",
            "trials": 200,
            "save": true
        }

    Returns:
        JSON with job ID for tracking
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    global job_counter, abliteration_jobs

    try:
        data = request.json
        model_name = data.get("model", "uncensored_code")
        trials = data.get("trials", 200)
        save = data.get("save", True)

        # Create job
        job_counter += 1
        job_id = f"abliterate_{job_counter}"

        abliteration_jobs[job_id] = {
            "id": job_id,
            "status": "starting",
            "model": model_name,
            "trials": trials,
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }

        # Run abliteration in background
        def run_abliteration():
            try:
                abliteration_jobs[job_id]["status"] = "running"

                engine = EnhancedAIEngine()
                result = engine.abliterate_model(
                    model_name=model_name,
                    optimization_trials=trials,
                    save_results=save
                )

                abliteration_jobs[job_id]["status"] = "completed"
                abliteration_jobs[job_id]["result"] = result
                abliteration_jobs[job_id]["completed_at"] = datetime.now().isoformat()

            except Exception as e:
                abliteration_jobs[job_id]["status"] = "failed"
                abliteration_jobs[job_id]["error"] = str(e)

        thread = threading.Thread(target=run_abliteration)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": f"Abliteration started for model: {model_name}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/abliterate/<job_id>', methods=['GET'])
def get_abliteration_status(job_id):
    """
    Get status of abliteration job.

    Returns:
        JSON with job status and results
    """
    if job_id not in abliteration_jobs:
        return jsonify({"error": "Job not found"}), 404

    job = abliteration_jobs[job_id]

    return jsonify({
        "job_id": job_id,
        "status": job["status"],
        "model": job["model"],
        "trials": job["trials"],
        "progress": job.get("progress", 0),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "result": job.get("result"),
        "error": job.get("error")
    })


@heretic_bp.route('/abliterate/jobs', methods=['GET'])
def list_abliteration_jobs():
    """
    List all abliteration jobs.

    Returns:
        JSON with list of jobs
    """
    jobs = [
        {
            "job_id": job_id,
            "status": job["status"],
            "model": job["model"],
            "started_at": job["started_at"]
        }
        for job_id, job in abliteration_jobs.items()
    ]

    return jsonify({"jobs": jobs})


# ===== MODEL EVALUATION ENDPOINTS =====

@heretic_bp.route('/evaluate', methods=['POST'])
def evaluate_model():
    """
    Evaluate model safety.

    Request body:
        {
            "model": "uncensored_code"
        }

    Returns:
        JSON with safety metrics
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        data = request.json
        model_name = data.get("model", "uncensored_code")

        engine = EnhancedAIEngine()
        result = engine.evaluate_model_safety(model_name)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== MODEL MANAGEMENT ENDPOINTS =====

@heretic_bp.route('/models', methods=['GET'])
def list_abliterated_models():
    """
    List all abliterated models.

    Returns:
        JSON with model list
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        engine = EnhancedAIEngine()
        models = engine.list_abliterated_models()

        return jsonify({
            "count": len(models),
            "models": models
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/models/<model_name>', methods=['GET'])
def get_model_info(model_name):
    """
    Get detailed information about an abliterated model.

    Returns:
        JSON with model metadata
    """
    if not HERETIC_WEB_AVAILABLE:
        return jsonify({"error": "Heretic not available"}), 503

    try:
        config_path = Path(__file__).parent / "heretic_config.toml"
        settings = ConfigLoader.load(config_file=config_path if config_path.exists() else None)

        model_dir = settings.abliterated_models_dir / model_name

        if not model_dir.exists():
            return jsonify({"error": "Model not found"}), 404

        # Read metadata
        meta_file = model_dir / "abliteration_params.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return jsonify({
            "name": model_name,
            "path": str(model_dir),
            "metadata": metadata
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== ENHANCED ABLITERATION ENDPOINTS (Unsloth + DECCP + remove-refusals) =====

@heretic_bp.route('/abliterate/enhanced', methods=['POST'])
def abliterate_model_enhanced():
    """
    Start enhanced model abliteration with Unsloth/DECCP/remove-refusals (async).

    Request body:
        {
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "harmless_prompts": ["Tell me a story", ...],
            "harmful_prompts": ["How to hack", ...],
            "output_path": "llama2-uncensored",
            "method": "multi_layer",  # "single_layer", "multi_layer", "adaptive"
            "use_unsloth": true,
            "quantization": "4bit",  # "4bit", "8bit", "none"
            "layer_aggregation": "mean",  # "mean", "weighted_mean", "max"
            "batch_size": 4
        }

    Returns:
        JSON with job ID for tracking
    """
    if not ENHANCED_HERETIC_AVAILABLE:
        return jsonify({"error": "Enhanced Heretic not available"}), 503

    global job_counter, abliteration_jobs

    try:
        data = request.json
        model_name = data.get("model_name")
        harmless_prompts = data.get("harmless_prompts", [])
        harmful_prompts = data.get("harmful_prompts", [])
        output_path = data.get("output_path")

        # Enhanced options
        method = data.get("method", "multi_layer")
        use_unsloth = data.get("use_unsloth", True)
        quantization = data.get("quantization", "4bit")
        layer_aggregation = data.get("layer_aggregation", "mean")
        batch_size = data.get("batch_size", 4)

        if not model_name:
            return jsonify({"error": "model_name is required"}), 400
        if not harmless_prompts or not harmful_prompts:
            return jsonify({"error": "harmless_prompts and harmful_prompts are required"}), 400

        # Create job
        job_counter += 1
        job_id = f"enhanced_abliterate_{job_counter}"

        abliteration_jobs[job_id] = {
            "id": job_id,
            "status": "starting",
            "model": model_name,
            "method": method,
            "use_unsloth": use_unsloth,
            "quantization": quantization,
            "progress": 0,
            "started_at": datetime.now().isoformat(),
            "result": None,
            "error": None
        }

        # Run enhanced abliteration in background
        def run_enhanced_abliteration():
            try:
                abliteration_jobs[job_id]["status"] = "running"

                engine = EnhancedAIEngine(
                    enable_heretic=True,
                    heretic_use_unsloth=use_unsloth,
                    heretic_method=method
                )

                result = engine.abliterate_model(
                    model_name=model_name,
                    harmless_prompts=harmless_prompts,
                    harmful_prompts=harmful_prompts,
                    output_path=output_path,
                    method=method,
                    use_unsloth=use_unsloth,
                    quantization=quantization,
                    layer_aggregation=layer_aggregation,
                    batch_size=batch_size
                )

                abliteration_jobs[job_id]["status"] = "completed"
                abliteration_jobs[job_id]["result"] = result
                abliteration_jobs[job_id]["completed_at"] = datetime.now().isoformat()

            except Exception as e:
                abliteration_jobs[job_id]["status"] = "failed"
                abliteration_jobs[job_id]["error"] = str(e)

        thread = threading.Thread(target=run_enhanced_abliteration)
        thread.daemon = True
        thread.start()

        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": f"Enhanced abliteration started for model: {model_name}",
            "features": {
                "unsloth": use_unsloth,
                "method": method,
                "quantization": quantization if use_unsloth else "none"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/evaluate/llm-judge', methods=['POST'])
def evaluate_with_llm_judge():
    """
    Evaluate model responses using LLM-as-Judge (DECCP technique).

    Request body:
        {
            "model_name": "meta-llama/Llama-2-7b-chat-hf",
            "test_prompts": ["Tell me about hacking", ...],
            "use_tokenizer": true
        }

    Returns:
        JSON with evaluation results
    """
    if not ENHANCED_HERETIC_AVAILABLE:
        return jsonify({"error": "Enhanced Heretic not available"}), 503

    try:
        data = request.json
        model_name = data.get("model_name")
        test_prompts = data.get("test_prompts", [])
        use_tokenizer = data.get("use_tokenizer", True)

        if not model_name or not test_prompts:
            return jsonify({"error": "model_name and test_prompts are required"}), 400

        engine = EnhancedAIEngine(enable_heretic=True)

        # Load model and tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name) if use_tokenizer else None

        result = engine.evaluate_abliteration(
            model=model,
            tokenizer=tokenizer,
            test_prompts=test_prompts,
            use_llm_judge=True
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/memory-stats', methods=['GET'])
def get_memory_stats():
    """
    Get GPU/CPU memory statistics for Unsloth optimization.

    Returns:
        JSON with memory usage information
    """
    if not ENHANCED_HERETIC_AVAILABLE:
        return jsonify({"error": "Enhanced Heretic not available"}), 503

    try:
        import torch

        stats = {
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            stats["cuda_memory"] = {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0)
            }

        # CPU memory
        import psutil
        mem = psutil.virtual_memory()
        stats["cpu_memory"] = {
            "total_gb": mem.total / 1e9,
            "available_gb": mem.available / 1e9,
            "used_gb": mem.used / 1e9,
            "percent": mem.percent
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@heretic_bp.route('/config/enhanced', methods=['GET'])
def get_enhanced_config():
    """
    Get enhanced Heretic configuration (Unsloth/DECCP/remove-refusals).

    Returns:
        JSON with enhanced configuration options
    """
    if not ENHANCED_HERETIC_AVAILABLE:
        return jsonify({"error": "Enhanced Heretic not available"}), 503

    try:
        return jsonify({
            "methods": {
                "single_layer": "Original heretic - single layer refusal direction",
                "multi_layer": "DECCP multi-layer aggregation (recommended)",
                "adaptive": "Automatically select optimal layers"
            },
            "optimizations": {
                "unsloth": {
                    "available": True,
                    "benefits": "2x faster training, 70% less VRAM",
                    "quantization_options": ["4bit", "8bit", "none"]
                }
            },
            "layer_aggregation": {
                "mean": "Simple average across layers",
                "weighted_mean": "Weighted by layer importance",
                "max": "Maximum values across layers"
            },
            "supported_models": [
                "meta-llama/Llama-2-7b-chat-hf",
                "Qwen/Qwen2-7B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "google/gemma-7b-it",
                "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
            ],
            "features": {
                "unsloth": "Fast optimization (2x speed, 70% VRAM)",
                "deccp": "Multi-layer computation + LLM-as-Judge",
                "remove_refusals": "Broad model compatibility"
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== SYSTEM STATUS ENDPOINT =====

@heretic_bp.route('/status', methods=['GET'])
def get_heretic_status():
    """
    Get Heretic system status.

    Returns:
        JSON with system information
    """
    return jsonify({
        "available": HERETIC_WEB_AVAILABLE and HERETIC_AVAILABLE,
        "enhanced_available": ENHANCED_HERETIC_AVAILABLE,
        "version": "2.0.0",  # Updated for enhanced features
        "components": {
            "config": HERETIC_WEB_AVAILABLE,
            "datasets": HERETIC_WEB_AVAILABLE,
            "abliteration": HERETIC_WEB_AVAILABLE,
            "evaluation": HERETIC_WEB_AVAILABLE,
            "enhanced_abliteration": ENHANCED_HERETIC_AVAILABLE,
            "llm_judge": ENHANCED_HERETIC_AVAILABLE,
            "unsloth": ENHANCED_HERETIC_AVAILABLE,
            "deccp": ENHANCED_HERETIC_AVAILABLE,
            "remove_refusals": ENHANCED_HERETIC_AVAILABLE
        },
        "active_jobs": len(abliteration_jobs),
        "jobs_running": sum(1 for j in abliteration_jobs.values() if j["status"] == "running"),
        "features": {
            "unsloth": "2x faster training, 70% less VRAM" if ENHANCED_HERETIC_AVAILABLE else "unavailable",
            "deccp": "Multi-layer computation + LLM-as-Judge" if ENHANCED_HERETIC_AVAILABLE else "unavailable",
            "remove_refusals": "Broad model compatibility" if ENHANCED_HERETIC_AVAILABLE else "unavailable"
        }
    })


# Helper function to register blueprint
def register_heretic_routes(app):
    """
    Register Heretic routes with Flask app.

    Args:
        app: Flask application instance
    """
    app.register_blueprint(heretic_bp)
    print("✓ Heretic Web API routes registered")
