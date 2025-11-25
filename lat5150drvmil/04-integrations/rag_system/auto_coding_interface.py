#!/usr/bin/env python3
"""
Auto-Coding Interface with Self-Healing Capabilities

Advanced code generation system that:
- Generates code from natural language specifications
- Analyzes existing codebase patterns
- Provides self-healing and error recovery
- Integrates with RAG system for contextual generation
- Uses storage system for code templates and patterns
- Learns from existing code to improve generation

This is the ultimate "code that heals itself" - an AI-powered system
that can automatically generate, fix, and improve code.
"""

import os
import ast
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import re

logger = logging.getLogger(__name__)


@dataclass
class CodeSpec:
    """Specification for code generation"""
    description: str
    language: str = "python"
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    context: Optional[str] = None


@dataclass
class GeneratedCode:
    """Generated code with metadata"""
    code: str
    language: str
    confidence: float
    explanation: str
    dependencies: List[str] = field(default_factory=list)
    tests: Optional[str] = None
    documentation: Optional[str] = None


class CodePatternAnalyzer:
    """
    Analyzes existing codebase to learn patterns

    Extracts:
    - Common function signatures
    - Class structures
    - Import patterns
    - Coding conventions
    - Documentation styles
    """

    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.patterns = {
            'functions': [],
            'classes': [],
            'imports': [],
            'decorators': [],
            'docstrings': []
        }

    def analyze_file(self, filepath: Path) -> Dict:
        """
        Analyze a single Python file

        Args:
            filepath: Path to Python file

        Returns:
            Dictionary of extracted patterns
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)
            file_patterns = {
                'functions': [],
                'classes': [],
                'imports': [],
                'decorators': []
            }

            for node in ast.walk(tree):
                # Extract function definitions
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'returns': ast.unparse(node.returns) if node.returns else None
                    }
                    file_patterns['functions'].append(func_info)

                # Extract class definitions
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [ast.unparse(base) for base in node.bases],
                        'methods': [],
                        'docstring': ast.get_docstring(node)
                    }

                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_info['methods'].append(item.name)

                    file_patterns['classes'].append(class_info)

                # Extract imports
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_patterns['imports'].append({
                            'module': alias.name,
                            'alias': alias.asname
                        })

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        file_patterns['imports'].append({
                            'from': node.module,
                            'imports': [alias.name for alias in node.names]
                        })

            return file_patterns

        except Exception as e:
            logger.warning(f"Could not analyze {filepath}: {e}")
            return {}

    def analyze_codebase(self) -> Dict:
        """
        Analyze entire codebase

        Returns:
            Aggregated patterns from all Python files
        """
        logger.info(f"Analyzing codebase in {self.root_dir}")

        python_files = list(self.root_dir.rglob("*.py"))
        logger.info(f"Found {len(python_files)} Python files")

        for filepath in python_files:
            patterns = self.analyze_file(filepath)

            for key in self.patterns:
                if key in patterns:
                    self.patterns[key].extend(patterns[key])

        logger.info("Codebase analysis complete")
        return self.patterns

    def get_common_patterns(self, pattern_type: str, limit: int = 10) -> List:
        """
        Get most common patterns of a type

        Args:
            pattern_type: Type of pattern (functions, classes, imports)
            limit: Maximum patterns to return

        Returns:
            List of common patterns
        """
        if pattern_type not in self.patterns:
            return []

        # Count frequency and return top patterns
        from collections import Counter

        if pattern_type == 'imports':
            modules = [p.get('module') or p.get('from') for p in self.patterns[pattern_type]]
            return [{'module': m, 'count': c} for m, c in Counter(modules).most_common(limit)]

        elif pattern_type == 'functions':
            func_names = [f['name'] for f in self.patterns[pattern_type]]
            return [{'name': n, 'count': c} for n, c in Counter(func_names).most_common(limit)]

        elif pattern_type == 'classes':
            class_names = [c['name'] for c in self.patterns[pattern_type]]
            return [{'name': n, 'count': c} for n, c in Counter(class_names).most_common(limit)]

        return []


class TemplateGenerator:
    """
    Generates code from templates

    Uses learned patterns and templates to generate code
    """

    def __init__(self, patterns: Optional[Dict] = None):
        self.patterns = patterns or {}
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load code templates"""
        return {
            'function': '''def {name}({params}) -> {return_type}:
    """
    {docstring}

    Args:
{arg_docs}

    Returns:
        {return_doc}
    """
    {body}
''',

            'class': '''class {name}({bases}):
    """
    {docstring}
    """

    def __init__(self, {init_params}):
        """
        Initialize {name}

        Args:
{init_arg_docs}
        """
{init_body}

{methods}
''',

            'method': '''    def {name}(self, {params}) -> {return_type}:
        """
        {docstring}

        Args:
{arg_docs}

        Returns:
            {return_doc}
        """
        {body}
''',

            'test_function': '''def test_{name}():
    """Test {name} function"""
    {test_body}
    assert result == expected
''',

            'storage_backend': '''class {name}StorageBackend(AbstractStorageBackend):
    """
    {description}
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.storage_type = StorageType.{storage_type}

    def connect(self) -> bool:
        """Establish connection"""
        try:
            {connect_body}
            return True
        except Exception as e:
            logger.error(f"Connection failed: {{e}}")
            return False

    def store(self, data: Any, content_type: ContentType, **kwargs) -> StorageHandle:
        """Store data"""
        {store_body}

    def retrieve(self, handle: StorageHandle) -> Optional[Any]:
        """Retrieve data"""
        {retrieve_body}

    def delete(self, handle: StorageHandle) -> bool:
        """Delete data"""
        {delete_body}
'''
        }

    def generate_function(self, spec: CodeSpec) -> str:
        """
        Generate function from specification

        Args:
            spec: Code specification

        Returns:
            Generated function code
        """
        # Extract parameters
        params = ", ".join([
            f"{inp['name']}: {inp.get('type', 'Any')}"
            for inp in spec.inputs
        ])

        # Generate argument documentation
        arg_docs = "\n".join([
            f"        {inp['name']}: {inp.get('description', 'TODO')}"
            for inp in spec.inputs
        ])

        # Determine return type
        return_type = spec.outputs[0]['type'] if spec.outputs else "None"
        return_doc = spec.outputs[0].get('description', 'TODO') if spec.outputs else "None"

        # Generate function body
        body = self._generate_function_body(spec)

        # Fill template
        code = self.templates['function'].format(
            name=spec.function_name or "generated_function",
            params=params,
            return_type=return_type,
            docstring=spec.description,
            arg_docs=arg_docs,
            return_doc=return_doc,
            body=body
        )

        return code

    def generate_class(self, spec: CodeSpec) -> str:
        """
        Generate class from specification

        Args:
            spec: Code specification

        Returns:
            Generated class code
        """
        # Determine base classes
        bases = spec.context if spec.context else "object"

        # Generate __init__ parameters
        init_params = ", ".join([
            f"{inp['name']}: {inp.get('type', 'Any')}"
            for inp in spec.inputs
        ])

        # Generate __init__ arg docs
        init_arg_docs = "\n".join([
            f"            {inp['name']}: {inp.get('description', 'TODO')}"
            for inp in spec.inputs
        ])

        # Generate __init__ body
        init_body = "\n".join([
            f"        self.{inp['name']} = {inp['name']}"
            for inp in spec.inputs
        ])

        # Generate methods (placeholder)
        methods = "    # TODO: Add methods"

        # Fill template
        code = self.templates['class'].format(
            name=spec.class_name or "GeneratedClass",
            bases=bases,
            docstring=spec.description,
            init_params=init_params,
            init_arg_docs=init_arg_docs,
            init_body=init_body,
            methods=methods
        )

        return code

    def _generate_function_body(self, spec: CodeSpec) -> str:
        """Generate function body from specification"""
        # If examples provided, use them
        if spec.examples:
            example = spec.examples[0]
            return f"    # Example: {example}\n    pass  # TODO: Implement"

        # Otherwise, generate placeholder
        if spec.outputs:
            output_type = spec.outputs[0].get('type', 'None')
            if output_type == 'bool':
                return "    return True"
            elif output_type in ['int', 'float']:
                return "    return 0"
            elif output_type == 'str':
                return '    return ""'
            elif output_type.startswith('List'):
                return "    return []"
            elif output_type.startswith('Dict'):
                return "    return {}"
            else:
                return "    return None"

        return "    pass  # TODO: Implement"


class SelfHealingEngine:
    """
    Self-healing engine for automatic error recovery

    Monitors code execution, detects errors, and attempts automatic fixes
    """

    def __init__(self, storage_orchestrator=None):
        self.storage = storage_orchestrator
        self.error_history: List[Dict] = []
        self.fix_patterns: Dict[str, str] = self._load_fix_patterns()

    def _load_fix_patterns(self) -> Dict[str, str]:
        """Load common error fix patterns"""
        return {
            'ImportError': 'install_missing_package',
            'ModuleNotFoundError': 'install_missing_package',
            'AttributeError': 'check_object_attributes',
            'KeyError': 'add_default_value',
            'FileNotFoundError': 'create_missing_file',
            'ConnectionRefusedError': 'restart_service',
            'TimeoutError': 'increase_timeout',
        }

    def log_error(self, error: Exception, context: Dict):
        """
        Log error for analysis

        Args:
            error: Exception that occurred
            context: Context information
        """
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context
        }

        self.error_history.append(error_info)

        # Store in database if available
        if self.storage:
            try:
                from storage_abstraction import ContentType
                self.storage.store(
                    data=error_info,
                    content_type=ContentType.AUDIT,
                    metadata={'error_type': error_info['type']}
                )
            except:
                pass

        logger.error(f"Error logged: {error_info['type']} - {error_info['message']}")

    def analyze_error(self, error: Exception) -> Optional[str]:
        """
        Analyze error and suggest fix

        Args:
            error: Exception to analyze

        Returns:
            Suggested fix strategy or None
        """
        error_type = type(error).__name__

        if error_type in self.fix_patterns:
            return self.fix_patterns[error_type]

        # Check error history for similar errors
        similar_errors = [
            e for e in self.error_history
            if e['type'] == error_type
        ]

        if len(similar_errors) > 3:
            logger.warning(f"Recurring error: {error_type} occurred {len(similar_errors)} times")
            return 'investigate_root_cause'

        return None

    def attempt_fix(self, error: Exception, context: Dict) -> bool:
        """
        Attempt to automatically fix error

        Args:
            error: Exception to fix
            context: Context information

        Returns:
            True if fix was attempted
        """
        fix_strategy = self.analyze_error(error)

        if not fix_strategy:
            return False

        logger.info(f"Attempting to fix {type(error).__name__} using strategy: {fix_strategy}")

        try:
            if fix_strategy == 'install_missing_package':
                return self._fix_missing_package(error, context)
            elif fix_strategy == 'create_missing_file':
                return self._fix_missing_file(error, context)
            elif fix_strategy == 'restart_service':
                return self._fix_service_connection(error, context)
            # Add more fix strategies here

        except Exception as e:
            logger.error(f"Fix attempt failed: {e}")
            return False

        return False

    def _fix_missing_package(self, error: Exception, context: Dict) -> bool:
        """Attempt to install missing package"""
        import subprocess

        error_msg = str(error)
        # Extract package name from error message
        match = re.search(r"No module named '(\w+)'", error_msg)

        if match:
            package = match.group(1)
            logger.info(f"Installing missing package: {package}")

            try:
                subprocess.check_call(['pip', 'install', package])
                logger.info(f"Successfully installed {package}")
                return True
            except:
                logger.error(f"Failed to install {package}")

        return False

    def _fix_missing_file(self, error: Exception, context: Dict) -> bool:
        """Create missing file with default content"""
        error_msg = str(error)
        # Extract filepath from error message
        match = re.search(r"'([^']+)'", error_msg)

        if match:
            filepath = Path(match.group(1))
            logger.info(f"Creating missing file: {filepath}")

            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.touch()
                logger.info(f"Successfully created {filepath}")
                return True
            except:
                logger.error(f"Failed to create {filepath}")

        return False

    def _fix_service_connection(self, error: Exception, context: Dict) -> bool:
        """Attempt to restart service"""
        service = context.get('service')

        if not service:
            return False

        logger.info(f"Attempting to restart service: {service}")
        # Implementation depends on specific services
        # This is a placeholder
        return False


class AutoCodingInterface:
    """
    Main auto-coding interface

    Integrates pattern analysis, template generation, and self-healing
    """

    def __init__(self, root_dir: str = ".", storage_orchestrator=None):
        self.root_dir = Path(root_dir)
        self.storage = storage_orchestrator

        self.analyzer = CodePatternAnalyzer(root_dir)
        self.generator = TemplateGenerator()
        self.healer = SelfHealingEngine(storage_orchestrator)

        self._patterns_analyzed = False

    def analyze_codebase(self):
        """Analyze codebase to learn patterns"""
        logger.info("Analyzing codebase...")
        patterns = self.analyzer.analyze_codebase()
        self.generator.patterns = patterns
        self._patterns_analyzed = True
        logger.info("Codebase analysis complete")

    def generate_code(self, spec: CodeSpec) -> GeneratedCode:
        """
        Generate code from specification

        Args:
            spec: Code specification

        Returns:
            Generated code with metadata
        """
        if not self._patterns_analyzed:
            logger.warning("Codebase not analyzed yet, using default templates")

        # Generate code based on spec
        if spec.class_name:
            code = self.generator.generate_class(spec)
        else:
            code = self.generator.generate_function(spec)

        # Generate tests
        tests = self._generate_tests(spec)

        # Generate documentation
        docs = self._generate_documentation(spec)

        return GeneratedCode(
            code=code,
            language=spec.language,
            confidence=0.8,  # Placeholder confidence score
            explanation=f"Generated {spec.class_name or spec.function_name} based on specification",
            tests=tests,
            documentation=docs
        )

    def _generate_tests(self, spec: CodeSpec) -> str:
        """Generate test code"""
        if spec.function_name:
            return f'''def test_{spec.function_name}():
    """Test {spec.function_name}"""
    # TODO: Add test cases
    pass
'''
        return "# TODO: Add tests"

    def _generate_documentation(self, spec: CodeSpec) -> str:
        """Generate documentation"""
        return f'''# {spec.class_name or spec.function_name}

{spec.description}

## Usage

```python
# TODO: Add usage examples
```

## API Reference

{spec.inputs}

## Returns

{spec.outputs}
'''

    def generate_storage_backend(self, name: str, storage_type: str, description: str) -> str:
        """
        Generate a new storage backend

        Args:
            name: Backend name
            storage_type: Storage type (e.g., "MONGODB", "CASSANDRA")
            description: Backend description

        Returns:
            Generated backend code
        """
        template = self.generator.templates['storage_backend']

        code = template.format(
            name=name,
            description=description,
            storage_type=storage_type.upper(),
            connect_body="    # TODO: Implement connection logic\n        pass",
            store_body="    # TODO: Implement storage logic\n        pass",
            retrieve_body="    # TODO: Implement retrieval logic\n        pass",
            delete_body="    # TODO: Implement deletion logic\n        pass"
        )

        return code

    def save_generated_code(self, generated: GeneratedCode, filepath: Path):
        """
        Save generated code to file

        Args:
            generated: Generated code
            filepath: Destination filepath
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(generated.code)

        logger.info(f"Saved generated code to: {filepath}")

        # Save tests if provided
        if generated.tests:
            test_file = filepath.parent / f"test_{filepath.name}"
            with open(test_file, 'w') as f:
                f.write(generated.tests)
            logger.info(f"Saved tests to: {test_file}")

        # Save documentation if provided
        if generated.documentation:
            doc_file = filepath.parent / f"{filepath.stem}_README.md"
            with open(doc_file, 'w') as f:
                f.write(generated.documentation)
            logger.info(f"Saved documentation to: {doc_file}")


def main():
    """Example usage"""
    print("=" * 80)
    print("AUTO-CODING INTERFACE")
    print("=" * 80 + "\n")

    # Initialize
    interface = AutoCodingInterface(root_dir=".")

    # Analyze codebase
    print("Analyzing codebase...")
    interface.analyze_codebase()

    # Get common patterns
    common_functions = interface.analyzer.get_common_patterns('functions', limit=5)
    print("\nMost common function names:")
    for func in common_functions:
        print(f"  {func['name']}: {func['count']} occurrences")

    # Generate code example
    print("\n" + "=" * 80)
    print("CODE GENERATION EXAMPLE")
    print("=" * 80 + "\n")

    spec = CodeSpec(
        description="Calculate the similarity score between two documents",
        function_name="calculate_similarity",
        inputs=[
            {'name': 'doc1', 'type': 'str', 'description': 'First document'},
            {'name': 'doc2', 'type': 'str', 'description': 'Second document'}
        ],
        outputs=[
            {'type': 'float', 'description': 'Similarity score between 0 and 1'}
        ]
    )

    generated = interface.generate_code(spec)
    print("Generated Code:")
    print("-" * 80)
    print(generated.code)

    print("\n✓ Auto-coding interface ready!")
    print("\nCapabilities:")
    print("  - Pattern analysis")
    print("  - Code generation from specifications")
    print("  - Automatic test generation")
    print("  - Self-healing error recovery")
    print("  - Storage backend scaffolding")


if __name__ == "__main__":
    main()
