#!/usr/bin/env python3
"""
Structured Data Extraction for LAT5150DRVMIL
Extract specific information in structured format
Based on Maharana et al. methodology
"""

import re
from typing import Dict, List, Optional
from rag_query import LAT5150RAG


class StructuredExtractor:
    """Extract structured information from documentation"""

    def __init__(self):
        self.rag = LAT5150RAG()

    def extract_system_info(self, system_name: str) -> Dict:
        """
        Extract structured information about a system/feature

        Returns dict with:
            - name
            - purpose
            - activation_steps
            - requirements
            - security_level
            - dependencies
        """
        # Query for the system
        results = self.rag.retriever.search(system_name, top_k=5)

        # Combine relevant contexts
        context = '\n\n'.join(
            result[0]['text']
            for result in results
            if result[1] > 0.1  # Relevance threshold
        )

        # Extract structured fields
        info = {
            'name': system_name,
            'purpose': self._extract_field(context, ['purpose', 'description', 'overview']),
            'activation_steps': self._extract_list(context, ['steps', 'activation', 'enable', 'setup']),
            'requirements': self._extract_list(context, ['requirements', 'prerequisites', 'needs']),
            'security_level': self._extract_field(context, ['security', 'classification', 'mil-spec']),
            'dependencies': self._extract_list(context, ['dependencies', 'requires', 'depends on']),
            'sources': [result[0]['metadata']['filepath'] for result in results[:3]]
        }

        return info

    def _extract_field(self, text: str, keywords: List[str]) -> str:
        """Extract a single field value"""
        text_lower = text.lower()

        for keyword in keywords:
            # Look for patterns like "Purpose: ..." or "Description: ..."
            pattern = rf"{keyword}:?\s*([^\n]{{50,200}})"
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return first relevant sentence
        sentences = text.split('.')
        for sent in sentences:
            if any(kw in sent.lower() for kw in keywords):
                return sent.strip()

        return "N/A"

    def _extract_list(self, text: str, keywords: List[str]) -> List[str]:
        """Extract a list of items"""
        items = []

        # Look for numbered lists
        numbered_pattern = r'\d+\.\s*([^\n]+)'
        numbered_matches = re.findall(numbered_pattern, text)

        # Look for bullet lists
        bullet_pattern = r'[-*]\s*([^\n]+)'
        bullet_matches = re.findall(bullet_pattern, text)

        # Combine and filter by keyword relevance
        all_matches = numbered_matches + bullet_matches

        for match in all_matches:
            if any(kw in match.lower() for kw in keywords):
                items.append(match.strip())

        return items[:5] if items else ["N/A"]

    def extract_configuration(self, component: str) -> Dict:
        """Extract configuration parameters"""
        results = self.rag.retriever.search(f"{component} configuration settings", top_k=3)

        context = '\n'.join(result[0]['text'] for result in results)

        # Extract key-value pairs
        config = {}

        # Pattern: KEY=VALUE or KEY: VALUE
        kv_pattern = r'(\w+(?:_\w+)*)\s*[:=]\s*([^\n,]+)'
        matches = re.findall(kv_pattern, context)

        for key, value in matches:
            config[key.strip()] = value.strip()

        return {
            'component': component,
            'parameters': config if config else {'status': 'No configuration found'},
            'sources': [result[0]['metadata']['filepath'] for result in results[:2]]
        }

    def batch_extract(self, systems: List[str]) -> List[Dict]:
        """Extract information for multiple systems"""
        results = []

        print(f"Extracting structured data for {len(systems)} systems...\n")

        for i, system in enumerate(systems, 1):
            print(f"[{i}/{len(systems)}] Extracting: {system}")
            info = self.extract_system_info(system)
            results.append(info)

        return results

    def save_to_json(self, data: List[Dict], output_file: str):
        """Save extracted data to JSON"""
        import json

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved to {output_file}")


def format_structured_output(info: Dict) -> str:
    """Format extracted info for display"""
    output = []
    output.append("="*60)
    output.append(f"System: {info['name']}")
    output.append("="*60)

    output.append(f"\nPurpose: {info['purpose']}")

    output.append("\nActivation Steps:")
    for step in info['activation_steps']:
        output.append(f"  • {step}")

    output.append("\nRequirements:")
    for req in info['requirements']:
        output.append(f"  • {req}")

    output.append(f"\nSecurity Level: {info['security_level']}")

    output.append("\nDependencies:")
    for dep in info['dependencies']:
        output.append(f"  • {dep}")

    output.append("\nSources:")
    for src in info['sources']:
        output.append(f"  📄 {src}")

    output.append("="*60)

    return '\n'.join(output)


if __name__ == '__main__':
    import sys

    extractor = StructuredExtractor()

    if len(sys.argv) > 1:
        # Extract specific system
        system_name = ' '.join(sys.argv[1:])
        print(f"Extracting information for: {system_name}\n")

        info = extractor.extract_system_info(system_name)
        print(format_structured_output(info))

    else:
        # Batch extraction example
        systems = [
            "DSMIL activation",
            "NPU modules",
            "APT41 security",
            "ZFS upgrade",
            "VAULT7 defense"
        ]

        results = extractor.batch_extract(systems)

        for info in results:
            print(format_structured_output(info))
            print()

        # Save to JSON
        extractor.save_to_json(results, 'rag_system/extracted_systems.json')
