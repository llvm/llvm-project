#!/usr/bin/env python3
"""
Tool Descriptor Generator

Interactive tool to create JSON descriptors for security tools.
Can use AI to help analyze tool capabilities and generate descriptors.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


class ToolDescriptorGenerator:
    """Generate tool descriptors interactively or with AI assistance"""

    def __init__(self, ai_model=None):
        """
        Initialize generator

        Args:
            ai_model: Optional AI model for assistance
        """
        self.ai_model = ai_model
        self.descriptors_dir = Path(__file__).parent / "tool_descriptors"

    def create_descriptor_interactive(self) -> Dict:
        """Create descriptor through interactive prompts"""
        print("\n🔧 Tool Descriptor Generator")
        print("=" * 70)
        print("Create a JSON descriptor for a security tool\n")

        descriptor = {}

        # Basic info
        descriptor["name"] = input("Tool name (e.g., 'nmap'): ").strip()
        descriptor["description"] = input("Description: ").strip()

        # Category
        print("\nCategories:")
        print("  1. reconnaissance")
        print("  2. vulnerability_scanning")
        print("  3. exploitation")
        print("  4. analysis")
        category = input("Select category (1-4): ").strip()
        categories = {
            "1": "reconnaissance",
            "2": "vulnerability_scanning",
            "3": "exploitation",
            "4": "analysis"
        }
        descriptor["category"] = categories.get(category, "reconnaissance")

        # Phases
        print("\nPhases (comma-separated):")
        print("  reconnaissance, analysis, reporting")
        phases = input("Phases: ").strip()
        descriptor["phase"] = [p.strip() for p in phases.split(",")]

        # Command
        descriptor["command"] = input("\nCommand to execute: ").strip()
        descriptor["required"] = input("Is this tool required? (y/n): ").lower() == 'y'

        # Arguments
        print("\nArguments (profiles):")
        print("Enter command arguments for different profiles.")
        print("Use {target} as placeholder for target URL/hostname")
        print("Use {port} for port, {output} for output file")
        print("\nExamples:")
        print("  basic: -sV -T4 {target}")
        print("  aggressive: -A -T4 -p- {target}")
        print("\nEnter 'done' when finished")

        args = {}
        while True:
            profile = input("\nProfile name (or 'done'): ").strip()
            if profile.lower() == 'done':
                break
            if profile:
                arg_str = input(f"  Arguments for '{profile}': ").strip()
                args[profile] = arg_str

        descriptor["args"] = args
        descriptor["default_profile"] = input("Default profile: ").strip()

        # Optional fields
        descriptor["timeout"] = int(input("\nTimeout (seconds, default 120): ").strip() or "120")
        descriptor["output_format"] = input("Output format (text/json/xml, default text): ").strip() or "text"
        descriptor["parse_output"] = input("Parse output? (y/n, default y): ").lower() != 'n'

        # Severity keywords
        print("\nSeverity keywords (optional):")
        print("Enter keywords that indicate different severity levels")
        print("Comma-separated, or press Enter to skip")

        severity_keywords = {}
        for level in ["critical", "high", "medium", "low", "info"]:
            keywords = input(f"  {level}: ").strip()
            if keywords:
                severity_keywords[level] = [k.strip() for k in keywords.split(",")]

        if severity_keywords:
            descriptor["severity_keywords"] = severity_keywords

        return descriptor

    def create_descriptor_with_ai(self, tool_name: str, tool_help: Optional[str] = None) -> Dict:
        """
        Create descriptor using AI analysis

        Args:
            tool_name: Name of the tool
            tool_help: Output from tool --help (optional)

        Returns:
            Dict descriptor
        """
        if not self.ai_model:
            print("⚠️  No AI model available, falling back to interactive mode")
            return self.create_descriptor_interactive()

        # Get tool help if not provided
        if not tool_help:
            tool_help = self._get_tool_help(tool_name)

        prompt = f"""Analyze this security tool and create a JSON descriptor for it.

Tool: {tool_name}
Help output:
{tool_help}

Create a JSON descriptor with these fields:
- name: Tool name
- description: Brief description of what it does
- category: One of: reconnaissance, vulnerability_scanning, exploitation, analysis
- phase: Array of phases it's used in (reconnaissance, analysis, reporting)
- command: Command to execute
- required: false
- args: Object with different profiles (basic, aggressive, stealth, etc.) and their arguments
  Use {{target}} as placeholder for target URL/hostname
- default_profile: Which profile to use by default
- timeout: Suggested timeout in seconds
- output_format: text, json, or xml
- parse_output: true if output should be parsed
- severity_keywords: Object mapping severity levels to keywords

Example format:
{{
  "name": "nmap",
  "description": "Network mapper for port scanning",
  "category": "reconnaissance",
  "phase": ["reconnaissance"],
  "command": "nmap",
  "required": false,
  "args": {{
    "basic": "-sV -T4 {{target}}",
    "aggressive": "-A -T4 -p- {{target}}"
  }},
  "default_profile": "basic",
  "timeout": 300,
  "output_format": "text",
  "parse_output": true,
  "severity_keywords": {{
    "critical": ["backdoor", "shell"],
    "high": ["vulnerable", "CVE-"]
  }}
}}

Generate the JSON descriptor:"""

        try:
            # Use AI model to generate descriptor
            # This would call the actual model
            print("🤖 Using AI to analyze tool...")
            print("⚠️  AI generation not yet implemented, using interactive mode")
            return self.create_descriptor_interactive()

        except Exception as e:
            print(f"⚠️  AI analysis failed: {e}")
            print("Falling back to interactive mode...")
            return self.create_descriptor_interactive()

    def _get_tool_help(self, tool_name: str) -> str:
        """Get tool help output"""
        try:
            # Try --help
            result = subprocess.run(
                [tool_name, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                return result.stdout[:2000]  # First 2000 chars

            # Try -h
            result = subprocess.run(
                [tool_name, "-h"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                return result.stdout[:2000]

            # Try man page
            result = subprocess.run(
                ["man", tool_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout:
                return result.stdout[:2000]

            return "No help available"

        except Exception as e:
            return f"Could not get help: {e}"

    def save_descriptor(self, descriptor: Dict) -> Path:
        """Save descriptor to JSON file"""
        filename = f"{descriptor['name']}.json"
        filepath = self.descriptors_dir / filename

        # Ensure directory exists
        self.descriptors_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        with open(filepath, 'w') as f:
            json.dump(descriptor, f, indent=2)

        return filepath

    def validate_descriptor(self, descriptor: Dict) -> bool:
        """Validate descriptor has required fields"""
        required_fields = ["name", "description", "category", "phase", "command", "args", "default_profile"]

        for field in required_fields:
            if field not in descriptor:
                print(f"❌ Missing required field: {field}")
                return False

        # Validate default_profile exists in args
        if descriptor["default_profile"] not in descriptor["args"]:
            print(f"❌ default_profile '{descriptor['default_profile']}' not found in args")
            return False

        return True


def main():
    """CLI interface"""
    print("\n🔧 Security Tool Descriptor Generator")
    print("=" * 70)

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python3 create_tool_descriptor.py interactive")
        print("  python3 create_tool_descriptor.py auto <tool_name>")
        print("  python3 create_tool_descriptor.py from-help <tool_name>")
        print("\nExamples:")
        print("  python3 create_tool_descriptor.py interactive")
        print("  python3 create_tool_descriptor.py auto nmap")
        print("  python3 create_tool_descriptor.py from-help gobuster")
        sys.exit(1)

    mode = sys.argv[1]
    generator = ToolDescriptorGenerator()

    if mode == "interactive":
        # Interactive mode
        descriptor = generator.create_descriptor_interactive()

    elif mode == "auto" and len(sys.argv) > 2:
        # Auto mode with AI
        tool_name = sys.argv[2]
        print(f"\n🤖 Analyzing tool: {tool_name}")
        descriptor = generator.create_descriptor_with_ai(tool_name)

    elif mode == "from-help" and len(sys.argv) > 2:
        # Auto mode with manual help
        tool_name = sys.argv[2]
        print(f"\n📖 Getting help for: {tool_name}")
        tool_help = generator._get_tool_help(tool_name)
        print(f"\nTool help output:\n{tool_help[:500]}...\n")
        descriptor = generator.create_descriptor_with_ai(tool_name, tool_help)

    else:
        print("❌ Invalid usage")
        sys.exit(1)

    # Validate
    print("\n✓ Validating descriptor...")
    if not generator.validate_descriptor(descriptor):
        print("❌ Validation failed")
        sys.exit(1)

    # Show preview
    print("\n📄 Generated descriptor:")
    print("=" * 70)
    print(json.dumps(descriptor, indent=2))
    print("=" * 70)

    # Confirm save
    save = input("\n💾 Save this descriptor? (y/n): ").lower()
    if save == 'y':
        filepath = generator.save_descriptor(descriptor)
        print(f"\n✅ Saved to: {filepath}")
        print("\n🔍 Test with:")
        print(f"   python3 security_agent.py <target>")
    else:
        print("\n❌ Descriptor not saved")


if __name__ == "__main__":
    main()
