#!/usr/bin/env python3
"""
DSMIL AI Engine - Device Configuration Utility
Auto-detects hardware and configures optimal settings for uncensored models

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
"""

import subprocess
import sys
import os
import json
from pathlib import Path

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'

def c(color):
    return color if sys.stdout.isatty() else ''

def run_command(cmd, timeout=5):
    """Run command and return output"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip()
    except:
        return False, ""

def detect_nvidia_gpu():
    """Detect NVIDIA GPU via nvidia-smi"""
    success, output = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if success and output:
        gpus = []
        for line in output.split('\n'):
            if line.strip():
                parts = line.split(',')
                if len(parts) == 2:
                    name = parts[0].strip()
                    memory = parts[1].strip()
                    gpus.append({"name": name, "memory": memory, "type": "NVIDIA CUDA"})
        return gpus
    return []

def detect_amd_gpu():
    """Detect AMD GPU via rocm-smi"""
    success, output = run_command("rocm-smi --showproductname")
    if success and "GPU" in output:
        gpus = []
        for line in output.split('\n'):
            if "GPU" in line and ":" in line:
                name = line.split(':', 1)[1].strip()
                gpus.append({"name": name, "memory": "Unknown", "type": "AMD ROCm"})
        return gpus
    return []

def detect_system_memory():
    """Get total system RAM"""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    kb = int(line.split()[1])
                    gb = kb / (1024 * 1024)
                    return round(gb, 1)
    except:
        pass
    return 0

def detect_cpu_info():
    """Get CPU information"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            lines = f.readlines()

        # Count cores
        cores = len([l for l in lines if l.startswith('processor')])

        # Get model name
        model = "Unknown"
        for line in lines:
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break

        return {"model": model, "cores": cores}
    except:
        return {"model": "Unknown", "cores": 0}

def check_ollama_running():
    """Check if Ollama is running"""
    success, _ = run_command("curl -s http://localhost:11434/api/tags")
    return success

def get_ollama_models():
    """Get list of installed Ollama models"""
    success, output = run_command("ollama list")
    if success:
        models = []
        for line in output.split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    return []

def recommend_device_config(gpus, ram_gb):
    """Recommend optimal device configuration"""
    config = {
        "device": "CPU",
        "recommendation": "",
        "settings": {}
    }

    if gpus:
        # GPU available
        gpu = gpus[0]
        gpu_type = gpu["type"]

        if "NVIDIA" in gpu_type:
            config["device"] = "NVIDIA CUDA"
            config["recommendation"] = f"Use GPU: {gpu['name']} ({gpu['memory']})"
            config["settings"] = {
                "CUDA_VISIBLE_DEVICES": "0",
                "OLLAMA_NUM_GPU": "1"
            }
        elif "AMD" in gpu_type:
            config["device"] = "AMD ROCm"
            config["recommendation"] = f"Use GPU: {gpu['name']}"
            config["settings"] = {
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",  # Common for ROCm
                "OLLAMA_NUM_GPU": "1"
            }
    else:
        # CPU only
        config["device"] = "CPU"
        if ram_gb >= 32:
            config["recommendation"] = "CPU with high RAM - can run 34B Q4_K_M models"
            config["settings"] = {
                "OLLAMA_NUM_GPU": "0",
                "OLLAMA_NUM_THREAD": "8"
            }
        elif ram_gb >= 16:
            config["recommendation"] = "CPU with moderate RAM - stick to 7-9B models"
            config["settings"] = {
                "OLLAMA_NUM_GPU": "0",
                "OLLAMA_NUM_THREAD": "4"
            }
        else:
            config["recommendation"] = "Low RAM - use smallest models only"
            config["settings"] = {
                "OLLAMA_NUM_GPU": "0",
                "OLLAMA_NUM_THREAD": "4"
            }

    return config

def generate_env_file(config):
    """Generate .env file for Ollama configuration"""
    env_file = Path.home() / ".dsmil" / "ollama.env"
    env_file.parent.mkdir(exist_ok=True)

    with open(env_file, 'w') as f:
        f.write("# DSMIL AI Engine - Ollama Configuration\n")
        f.write("# Auto-generated by configure_device.py\n\n")

        for key, value in config["settings"].items():
            f.write(f"export {key}={value}\n")

        f.write("\n# Apply with: source ~/.dsmil/ollama.env\n")

    return env_file

def print_status():
    """Print current system status"""
    print(f"\n{c(Colors.BOLD)}{c(Colors.CYAN)}DSMIL AI - Device Configuration{c(Colors.RESET)}")
    print(f"{c(Colors.GRAY)}Hardware detection and optimization{c(Colors.RESET)}\n")

    # Check Ollama
    print(f"{c(Colors.BOLD)}Ollama Status{c(Colors.RESET)}")
    ollama_running = check_ollama_running()
    if ollama_running:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} Ollama is running")

        models = get_ollama_models()
        uncensored_models = [m for m in models if 'uncensored' in m.lower() or 'wizard' in m.lower()]

        if uncensored_models:
            print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} {len(uncensored_models)} uncensored model(s) installed")
            for model in uncensored_models[:3]:  # Show first 3
                print(f"    → {model}")
        else:
            print(f"  {c(Colors.YELLOW)}⚠{c(Colors.RESET)} No uncensored models found")
            print(f"    Run: ./setup_uncensored_models.sh")
    else:
        print(f"  {c(Colors.RED)}✗{c(Colors.RESET)} Ollama is not running")
        print(f"    Start with: systemctl start ollama")

    print()

    # CPU Info
    cpu = detect_cpu_info()
    print(f"{c(Colors.BOLD)}CPU{c(Colors.RESET)}")
    print(f"  Model: {cpu['model']}")
    print(f"  Cores: {cpu['cores']}")
    print()

    # RAM
    ram_gb = detect_system_memory()
    print(f"{c(Colors.BOLD)}RAM{c(Colors.RESET)}")
    print(f"  Total: {ram_gb} GB")

    # Recommend models based on RAM
    if ram_gb >= 64:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} Can run all models (including 70B)")
    elif ram_gb >= 32:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} Can run 34B models with Q4_K_M quantization")
    elif ram_gb >= 16:
        print(f"  {c(Colors.YELLOW)}⚠{c(Colors.RESET)} Recommended: 7-9B models only")
    else:
        print(f"  {c(Colors.RED)}✗{c(Colors.RESET)} Low RAM - use smallest models")
    print()

    # GPU Detection
    print(f"{c(Colors.BOLD)}GPU{c(Colors.RESET)}")
    nvidia_gpus = detect_nvidia_gpu()
    amd_gpus = detect_amd_gpu()

    gpus = nvidia_gpus + amd_gpus

    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} GPU {i}: {gpu['name']} ({gpu['type']})")
            if gpu['memory'] != "Unknown":
                print(f"    Memory: {gpu['memory']}")
    else:
        print(f"  {c(Colors.YELLOW)}⚠{c(Colors.RESET)} No GPU detected - will use CPU")

    print()

    # Recommendation
    config = recommend_device_config(gpus, ram_gb)
    print(f"{c(Colors.BOLD)}Recommendation{c(Colors.RESET)}")
    print(f"  Device: {config['device']}")
    print(f"  {config['recommendation']}")
    print()

    # Optimal models
    print(f"{c(Colors.BOLD)}Optimal Models for Your Hardware{c(Colors.RESET)}")

    if gpus and ram_gb >= 16:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} wizardlm-uncensored-codellama:34b-q4_K_M (RECOMMENDED)")
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} wizardcoder:34b-python-q4_K_M")
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} yi-coder:9b-chat-q4_K_M")
    elif ram_gb >= 32:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} wizardlm-uncensored-codellama:34b-q4_K_M (RECOMMENDED)")
        print(f"  {c(Colors.YELLOW)}⚠{c(Colors.RESET)} Inference will be slower on CPU")
    elif ram_gb >= 16:
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} mistral:7b-instruct-uncensored (RECOMMENDED)")
        print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} yi-coder:9b-chat-q4_K_M")
    else:
        print(f"  {c(Colors.YELLOW)}⚠{c(Colors.RESET)} deepseek-r1:1.5b (fast model only)")

    print()

    return config, gpus, ram_gb

def apply_config(config):
    """Apply device configuration"""
    print(f"{c(Colors.BOLD)}Applying Configuration{c(Colors.RESET)}")

    # Generate env file
    env_file = generate_env_file(config)
    print(f"  {c(Colors.GREEN)}✓{c(Colors.RESET)} Configuration saved to {env_file}")

    print()
    print(f"{c(Colors.BOLD)}Next Steps{c(Colors.RESET)}")
    print(f"  1. Apply settings:")
    print(f"     source {env_file}")
    print(f"     systemctl restart ollama")
    print()
    print(f"  2. Install recommended models:")
    print(f"     ./setup_uncensored_models.sh")
    print()
    print(f"  3. Launch AI interface:")
    print(f"     python3 ai-tui-default")
    print()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        config, gpus, ram_gb = print_status()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--apply":
        config, gpus, ram_gb = print_status()
        print()
        apply_config(config)
        sys.exit(0)

    # Interactive mode
    config, gpus, ram_gb = print_status()

    print(f"{c(Colors.BOLD)}Actions{c(Colors.RESET)}")
    print(f"  1. Apply recommended configuration")
    print(f"  2. Install uncensored models")
    print(f"  3. Test inference")
    print(f"  4. Exit")
    print()

    choice = input(f"{c(Colors.GRAY)}Choose [1-4]: {c(Colors.RESET)}").strip()

    if choice == "1":
        apply_config(config)

    elif choice == "2":
        print()
        print(f"{c(Colors.CYAN)}→ Launching model installer...{c(Colors.RESET)}")
        os.system("./setup_uncensored_models.sh")

    elif choice == "3":
        if not check_ollama_running():
            print(f"{c(Colors.RED)}✗ Ollama is not running{c(Colors.RESET)}")
            sys.exit(1)

        print()
        print(f"{c(Colors.CYAN)}→ Testing inference...{c(Colors.RESET)}")
        print()

        # Test with uncensored model
        test_prompt = "Write a Python function to reverse a string"
        print(f"Prompt: {test_prompt}")
        print()

        result = subprocess.run(
            ['ollama', 'run', 'wizardlm-uncensored-codellama:34b-q4_K_M', test_prompt],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print(result.stdout)
            print()
            print(f"{c(Colors.GREEN)}✓ Inference test successful{c(Colors.RESET)}")
        else:
            print(f"{c(Colors.RED)}✗ Inference test failed{c(Colors.RESET)}")
            print(result.stderr)

    elif choice == "4":
        sys.exit(0)

    else:
        print(f"{c(Colors.RED)}Invalid choice{c(Colors.RESET)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
