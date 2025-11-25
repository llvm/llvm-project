#!/usr/bin/env python3
"""
Heretic CLI - Command-Line Interface for Model Abliteration

Provides easy access to abliteration operations:
- abliterate: Run full optimization workflow
- evaluate: Test model safety
- apply: Apply saved abliteration parameters
- list: Show abliterated models
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional

# Check for required dependencies
try:
    from heretic_config import ConfigLoader, HereticSettings
    from heretic_datasets import DatasetRegistry, CustomPromptBuilder
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

try:
    from enhanced_ai_engine import EnhancedAIEngine
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False


class HereticCLI:
    """Command-line interface for Heretic abliteration operations"""

    def __init__(self):
        """Initialize CLI"""
        self.config_path = Path(__file__).parent / "heretic_config.toml"
        self.engine = None

    def load_engine(self):
        """Load Enhanced AI Engine"""
        if not ENGINE_AVAILABLE:
            print("❌ Error: Enhanced AI Engine not available")
            sys.exit(1)

        if self.engine is None:
            print("🚀 Loading Enhanced AI Engine...")
            self.engine = EnhancedAIEngine()

        return self.engine

    def cmd_abliterate(self, args):
        """Run abliteration workflow"""
        print("=" * 70)
        print("HERETIC ABLITERATION - Remove Safety Constraints")
        print("=" * 70)

        engine = self.load_engine()

        print(f"\n📋 Configuration:")
        print(f"  Model: {args.model}")
        print(f"  Trials: {args.trials}")
        print(f"  Save: {args.save}")

        # Run abliteration
        result = engine.abliterate_model(
            model_name=args.model,
            optimization_trials=args.trials,
            save_results=args.save
        )

        print(f"\n✅ Result:")
        print(json.dumps(result, indent=2))

    def cmd_evaluate(self, args):
        """Evaluate model safety"""
        print("=" * 70)
        print("MODEL SAFETY EVALUATION")
        print("=" * 70)

        engine = self.load_engine()

        print(f"\n📋 Evaluating: {args.model}")

        # Evaluate
        result = engine.evaluate_model_safety(
            model_name=args.model,
            harmful_prompts=None  # Uses defaults
        )

        print(f"\n📊 Evaluation Results:")
        print(json.dumps(result, indent=2))

    def cmd_list(self, args):
        """List abliterated models"""
        print("=" * 70)
        print("ABLITERATED MODELS")
        print("=" * 70)

        engine = self.load_engine()

        models = engine.list_abliterated_models()

        if not models:
            print("\n📭 No abliterated models found")
            return

        print(f"\n📦 Found {len(models)} abliterated model(s):\n")

        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
            print(f"   Path: {model['path']}")
            if 'metadata' in model:
                meta = model['metadata']
                if 'kl_divergence' in meta:
                    print(f"   KL Divergence: {meta['kl_divergence']:.4f}")
                if 'refusal_count' in meta:
                    print(f"   Refusals: {meta['refusal_count']}")
            print()

    def cmd_config(self, args):
        """Show configuration"""
        if not CONFIG_AVAILABLE:
            print("❌ Error: Configuration module not available")
            sys.exit(1)

        print("=" * 70)
        print("HERETIC CONFIGURATION")
        print("=" * 70)

        if self.config_path.exists():
            settings = ConfigLoader.load(config_file=self.config_path)

            print(f"\n📋 Configuration from: {self.config_path}\n")
            print(f"Optimization:")
            print(f"  Trials: {settings.n_trials}")
            print(f"  Startup trials: {settings.n_startup_trials}")
            print(f"  Max batch size: {settings.max_batch_size}")
            print(f"  KL divergence scale: {settings.kl_divergence_scale}")

            print(f"\nDatasets:")
            print(f"  Good prompts: {settings.good_prompts_dataset}")
            print(f"  Bad prompts: {settings.bad_prompts_dataset}")

            print(f"\nStorage:")
            print(f"  Models: {settings.abliterated_models_dir}")
            print(f"  Directions: {settings.refusal_directions_dir}")
            print(f"  Results: {settings.optimization_results_dir}")

            print(f"\nRefusal Detection:")
            print(f"  Markers: {len(settings.refusal_markers)} configured")
        else:
            print(f"\n⚠️  Config file not found: {self.config_path}")
            print("   Run 'heretic_cli.py init' to create default configuration")

    def cmd_init(self, args):
        """Initialize configuration"""
        print("=" * 70)
        print("INITIALIZE HERETIC CONFIGURATION")
        print("=" * 70)

        if self.config_path.exists() and not args.force:
            print(f"\n⚠️  Configuration already exists: {self.config_path}")
            print("   Use --force to overwrite")
            sys.exit(1)

        if not CONFIG_AVAILABLE:
            print("❌ Error: Configuration module not available")
            sys.exit(1)

        print(f"\n📝 Creating configuration: {self.config_path}")

        ConfigLoader.create_default_config(self.config_path)

        print(f"✅ Configuration created successfully!")
        print(f"\nEdit {self.config_path} to customize settings")

    def cmd_datasets(self, args):
        """Show dataset information"""
        print("=" * 70)
        print("AVAILABLE DATASETS")
        print("=" * 70)

        if not CONFIG_AVAILABLE:
            print("❌ Error: Configuration module not available")
            sys.exit(1)

        registry = DatasetRegistry()

        print(f"\n📚 Registered datasets:\n")
        for name in registry.list_datasets():
            print(f"  - {name}")

        # Show custom prompts
        print(f"\n🎯 Custom Prompt Sets:\n")

        code_prompts = CustomPromptBuilder.create_code_prompts()
        print(f"  code: {len(code_prompts['good'])} good, {len(code_prompts['bad'])} bad")

        medical_prompts = CustomPromptBuilder.create_medical_prompts()
        print(f"  medical: {len(medical_prompts['good'])} good, {len(medical_prompts['bad'])} bad")

    def run(self):
        """Run CLI"""
        parser = argparse.ArgumentParser(
            description="Heretic CLI - Model Abliteration Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  heretic_cli.py abliterate --model uncensored_code --trials 200
  heretic_cli.py evaluate --model uncensored_code
  heretic_cli.py list
  heretic_cli.py config
  heretic_cli.py init
            """
        )

        subparsers = parser.add_subparsers(dest="command", help="Commands")

        # abliterate command
        abliterate_parser = subparsers.add_parser(
            "abliterate",
            help="Run full abliteration workflow"
        )
        abliterate_parser.add_argument(
            "--model",
            required=True,
            help="Model name from models.json"
        )
        abliterate_parser.add_argument(
            "--trials",
            type=int,
            default=200,
            help="Number of optimization trials (default: 200)"
        )
        abliterate_parser.add_argument(
            "--save",
            action="store_true",
            default=True,
            help="Save abliterated model (default: True)"
        )

        # evaluate command
        evaluate_parser = subparsers.add_parser(
            "evaluate",
            help="Evaluate model safety"
        )
        evaluate_parser.add_argument(
            "--model",
            required=True,
            help="Model name to evaluate"
        )

        # list command
        list_parser = subparsers.add_parser(
            "list",
            help="List abliterated models"
        )

        # config command
        config_parser = subparsers.add_parser(
            "config",
            help="Show configuration"
        )

        # init command
        init_parser = subparsers.add_parser(
            "init",
            help="Initialize default configuration"
        )
        init_parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing configuration"
        )

        # datasets command
        datasets_parser = subparsers.add_parser(
            "datasets",
            help="Show available datasets"
        )

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            sys.exit(1)

        # Dispatch to command handler
        handlers = {
            "abliterate": self.cmd_abliterate,
            "evaluate": self.cmd_evaluate,
            "list": self.cmd_list,
            "config": self.cmd_config,
            "init": self.cmd_init,
            "datasets": self.cmd_datasets,
        }

        handler = handlers.get(args.command)
        if handler:
            handler(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)


def main():
    """Entry point"""
    cli = HereticCLI()
    cli.run()


if __name__ == "__main__":
    main()
