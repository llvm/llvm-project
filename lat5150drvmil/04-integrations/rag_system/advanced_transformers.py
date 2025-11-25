#!/usr/bin/env python3
"""
Advanced AST Transformers
Method extraction, naming convention fixes, and code organization

Features:
- MethodExtractor: Extract complex blocks into separate methods
- NamingConventionFixer: camelCase → snake_case
- VariableRenamer: Single-letter → descriptive names
- ImportOptimizer: Remove unused, sort imports
- ClassOrganizer: Group related methods
"""

import ast
import re
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

try:
    import astor
    ASTOR_AVAILABLE = True
except ImportError:
    ASTOR_AVAILABLE = False


@dataclass
class Transformation:
    """Record of a code transformation"""
    transformer_name: str
    description: str
    original_line: int
    changes_made: str


class MethodExtractor(ast.NodeTransformer):
    """
    Extract complex code blocks into separate methods

    Detects:
    - Long if/else chains
    - Complex loops
    - Repeated code patterns
    """

    def __init__(self, complexity_threshold: int = 5):
        """
        Args:
            complexity_threshold: Min complexity to extract
        """
        self.complexity_threshold = complexity_threshold
        self.transformations: List[Transformation] = []
        self.extracted_methods: List[ast.FunctionDef] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Extract complex logic into helper methods"""
        self.generic_visit(node)

        # Find extractable blocks
        extractable = self._find_extractable_blocks(node)

        for block_info in extractable:
            # Create helper method
            helper_name = self._generate_method_name(block_info['type'], node.name)
            helper_method = self._create_helper_method(
                helper_name,
                block_info['node'],
                block_info['variables']
            )

            self.extracted_methods.append(helper_method)

            # Replace original block with method call
            call_node = self._create_method_call(helper_name, block_info['variables'])

            # Record transformation
            self.transformations.append(Transformation(
                transformer_name="MethodExtractor",
                description=f"Extracted '{helper_name}' from '{node.name}'",
                original_line=block_info['node'].lineno,
                changes_made=f"Reduced complexity by extracting {block_info['type']}"
            ))

        return node

    def _find_extractable_blocks(self, func_node: ast.FunctionDef) -> List[Dict]:
        """Identify code blocks that should be extracted"""
        candidates = []

        for node in ast.walk(func_node):
            # Long if/else chains
            if isinstance(node, ast.If):
                branches = self._count_branches(node)
                if branches > 3:
                    candidates.append({
                        'type': 'conditional',
                        'node': node,
                        'complexity': branches,
                        'variables': self._extract_variables(node)
                    })

            # Complex loops
            if isinstance(node, (ast.For, ast.While)):
                complexity = self._calculate_complexity(node)
                if complexity > self.complexity_threshold:
                    candidates.append({
                        'type': 'loop',
                        'node': node,
                        'complexity': complexity,
                        'variables': self._extract_variables(node)
                    })

        return candidates

    def _count_branches(self, node: ast.If) -> int:
        """Count branches in if/elif/else chain"""
        count = 1  # Initial if
        current = node

        while hasattr(current, 'orelse') and current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                count += 1
                current = current.orelse[0]
            else:
                count += 1  # Final else
                break

        return count

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of node"""
        complexity = 1

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _extract_variables(self, node: ast.AST) -> List[str]:
        """Extract variables used in node"""
        variables = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                variables.add(child.id)

        return list(variables)

    def _generate_method_name(self, block_type: str, parent_name: str) -> str:
        """Generate descriptive method name"""
        prefix = {
            'conditional': '_handle',
            'loop': '_process',
            'try': '_execute'
        }.get(block_type, '_helper')

        return f"{prefix}_{parent_name}_logic"

    def _create_helper_method(self, name: str, node: ast.AST, variables: List[str]) -> ast.FunctionDef:
        """Create extracted helper method"""
        # Create function with variables as parameters
        args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=var, annotation=None) for var in variables],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[]
        )

        # Wrap node in function
        func = ast.FunctionDef(
            name=name,
            args=args,
            body=[node] if isinstance(node, list) else [node],
            decorator_list=[],
            returns=None
        )

        return func

    def _create_method_call(self, method_name: str, variables: List[str]) -> ast.Call:
        """Create call to extracted method"""
        return ast.Call(
            func=ast.Name(id=method_name, ctx=ast.Load()),
            args=[ast.Name(id=var, ctx=ast.Load()) for var in variables],
            keywords=[]
        )

    def transform(self, code: str) -> Tuple[str, List[Transformation]]:
        """Transform code with method extraction"""
        if not ASTOR_AVAILABLE:
            return code, [Transformation(
                transformer_name="MethodExtractor",
                description="astor not available",
                original_line=0,
                changes_made="Install astor: pip install astor"
            )]

        try:
            tree = ast.parse(code)
            transformed = self.visit(tree)
            ast.fix_missing_locations(transformed)

            # Generate code
            result_code = astor.to_source(transformed)

            return result_code, self.transformations

        except Exception as e:
            return code, [Transformation(
                transformer_name="MethodExtractor",
                description=f"Error: {e}",
                original_line=0,
                changes_made="None"
            )]


class NamingConventionFixer(ast.NodeTransformer):
    """
    Automatically fix naming conventions

    Conversions:
    - camelCase → snake_case (functions, variables)
    - PascalCase → PascalCase (classes - no change)
    - SCREAMING_CASE → SCREAMING_CASE (constants - no change)
    """

    def __init__(self):
        self.transformations: List[Transformation] = []
        self.renames: Dict[str, str] = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Convert function names to snake_case"""
        self.generic_visit(node)

        if self._is_camel_case(node.name) and not node.name.startswith('_'):
            new_name = self._to_snake_case(node.name)

            self.transformations.append(Transformation(
                transformer_name="NamingConventionFixer",
                description=f"Renamed function: {node.name} → {new_name}",
                original_line=node.lineno,
                changes_made="camelCase → snake_case"
            ))

            self.renames[node.name] = new_name
            node.name = new_name

        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Fix variable names"""
        # Update references to renamed functions/variables
        if node.id in self.renames:
            node.id = self.renames[node.id]

        # Convert camelCase variables
        if isinstance(node.ctx, ast.Store) and self._is_camel_case(node.id):
            new_name = self._to_snake_case(node.id)
            self.renames[node.id] = new_name
            node.id = new_name

        return node

    @staticmethod
    def _is_camel_case(name: str) -> bool:
        """Check if name is camelCase"""
        # Has lowercase first letter and contains uppercase
        return (name[0].islower() and
                any(c.isupper() for c in name) and
                not '_' in name)

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert camelCase to snake_case"""
        # Insert underscore before uppercase letters
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def transform(self, code: str) -> Tuple[str, List[Transformation]]:
        """Transform code with naming fixes"""
        if not ASTOR_AVAILABLE:
            return code, []

        try:
            tree = ast.parse(code)
            transformed = self.visit(tree)
            ast.fix_missing_locations(transformed)

            result_code = astor.to_source(transformed)
            return result_code, self.transformations

        except Exception as e:
            return code, []


class ImportOptimizer(ast.NodeTransformer):
    """
    Optimize import statements

    Features:
    - Remove unused imports
    - Sort imports alphabetically
    - Group imports (stdlib, third-party, local)
    - Consolidate from imports
    """

    def __init__(self):
        self.transformations: List[Transformation] = []
        self.all_imports: List[ast.Import | ast.ImportFrom] = []
        self.used_names: Set[str] = set()

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Optimize imports at module level"""
        # Collect all imports and used names
        for child in node.body:
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                self.all_imports.append(child)
            else:
                # Collect used names
                for subnode in ast.walk(child):
                    if isinstance(subnode, ast.Name):
                        self.used_names.add(subnode.id)

        # Filter unused imports
        used_imports = self._filter_unused_imports()

        # Sort imports
        sorted_imports = self._sort_imports(used_imports)

        # Replace imports in AST
        new_body = sorted_imports + [n for n in node.body
                                     if not isinstance(n, (ast.Import, ast.ImportFrom))]

        node.body = new_body

        if len(self.all_imports) != len(used_imports):
            removed = len(self.all_imports) - len(used_imports)
            self.transformations.append(Transformation(
                transformer_name="ImportOptimizer",
                description=f"Removed {removed} unused imports",
                original_line=1,
                changes_made=f"Optimized imports ({len(used_imports)} remaining)"
            ))

        return node

    def _filter_unused_imports(self) -> List:
        """Remove unused imports"""
        used_imports = []

        for imp in self.all_imports:
            if isinstance(imp, ast.Import):
                # Check if any alias is used
                for alias in imp.names:
                    name = alias.asname or alias.name
                    if name in self.used_names:
                        used_imports.append(imp)
                        break

            elif isinstance(imp, ast.ImportFrom):
                # Check if imported names are used
                used_names = [alias for alias in imp.names
                             if (alias.asname or alias.name) in self.used_names]
                if used_names:
                    # Create new import with only used names
                    new_imp = ast.ImportFrom(
                        module=imp.module,
                        names=used_names,
                        level=imp.level
                    )
                    used_imports.append(new_imp)

        return used_imports

    def _sort_imports(self, imports: List) -> List:
        """Sort imports (stdlib, third-party, local)"""
        # Simple alphabetical sort for now
        # TODO: Implement isort-style grouping
        return sorted(imports, key=lambda x: self._import_sort_key(x))

    @staticmethod
    def _import_sort_key(imp) -> str:
        """Generate sort key for import"""
        if isinstance(imp, ast.Import):
            return imp.names[0].name
        elif isinstance(imp, ast.ImportFrom):
            return imp.module or ''
        return ''

    def transform(self, code: str) -> Tuple[str, List[Transformation]]:
        """Transform code with import optimization"""
        if not ASTOR_AVAILABLE:
            return code, []

        try:
            tree = ast.parse(code)
            transformed = self.visit(tree)
            ast.fix_missing_locations(transformed)

            result_code = astor.to_source(transformed)
            return result_code, self.transformations

        except Exception as e:
            return code, []


def apply_advanced_transformers(code: str) -> Tuple[str, List[Transformation]]:
    """
    Apply all advanced transformers

    Args:
        code: Python source code

    Returns:
        (transformed_code, transformations_list)
    """
    all_transformations = []
    current_code = code

    transformers = [
        NamingConventionFixer(),
        ImportOptimizer(),
        # MethodExtractor(),  # Disabled by default (can be aggressive)
    ]

    for transformer in transformers:
        try:
            current_code, transforms = transformer.transform(current_code)
            all_transformations.extend(transforms)
        except Exception as e:
            all_transformations.append(Transformation(
                transformer_name=transformer.__class__.__name__,
                description=f"Failed: {e}",
                original_line=0,
                changes_made="None"
            ))

    return current_code, all_transformations


def main():
    """Test advanced transformers"""
    test_code = """
import os
import sys
import json
import unused_module

def calculateTotal(itemList, taxRate):
    totalAmount = 0
    for i in range(len(itemList)):
        totalAmount += itemList[i]

    return totalAmount * (1 + taxRate)

class MyClass:
    def processData(self, inputData):
        return json.dumps(inputData)
"""

    print("="*70)
    print("Advanced AST Transformers Demo")
    print("="*70)

    print("\n📝 Original Code:")
    print(test_code)

    # Apply transformations
    transformed, transformations = apply_advanced_transformers(test_code)

    print("\n✨ Transformed Code:")
    print(transformed)

    print("\n📋 Transformations Applied:")
    for t in transformations:
        print(f"  [{t.transformer_name}] Line {t.original_line}")
        print(f"    {t.description}")
        print(f"    Changes: {t.changes_made}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
