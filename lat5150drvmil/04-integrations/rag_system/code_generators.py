#!/usr/bin/env python3
"""
Code Generators
Automatic generation of documentation, tests, and boilerplate code

Features:
- Generate docstrings for functions/classes
- Generate unit tests (pytest/unittest)
- Generate type stubs (.pyi files)
- Generate README documentation
"""

import ast
import inspect
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GeneratedDoc:
    """Generated documentation"""
    target: str  # Function/class name
    docstring: str
    line: int


@dataclass
class GeneratedTest:
    """Generated test case"""
    test_name: str
    test_code: str
    framework: str  # 'pytest' or 'unittest'


class DocumentationGenerator:
    """
    Automatically generate docstrings for functions and classes
    """

    def __init__(self, style: str = 'google'):
        """
        Args:
            style: Docstring style ('google', 'numpy', or 'sphinx')
        """
        self.style = style

    def generate_function_docstring(self, func_node: ast.FunctionDef, code: str) -> str:
        """
        Generate docstring for a function

        Args:
            func_node: AST node for function
            code: Full source code

        Returns:
            Generated docstring
        """
        # Extract function signature information
        args = [arg.arg for arg in func_node.args.args if arg.arg != 'self']
        returns = func_node.returns is not None

        # Check if function has return statement
        has_return = any(isinstance(node, ast.Return) and node.value is not None
                        for node in ast.walk(func_node))

        if self.style == 'google':
            return self._google_style_docstring(func_node.name, args, has_return)
        elif self.style == 'numpy':
            return self._numpy_style_docstring(func_node.name, args, has_return)
        else:  # sphinx
            return self._sphinx_style_docstring(func_node.name, args, has_return)

    def _google_style_docstring(self, func_name: str, args: List[str], has_return: bool) -> str:
        """Generate Google-style docstring"""
        doc_lines = [
            f'"""',
            f'{self._generate_summary(func_name)}',
            ''
        ]

        if args:
            doc_lines.append('Args:')
            for arg in args:
                doc_lines.append(f'    {arg}: Description of {arg}')
            doc_lines.append('')

        if has_return:
            doc_lines.append('Returns:')
            doc_lines.append('    Description of return value')
            doc_lines.append('')

        doc_lines.append('"""')

        return '\n    '.join(doc_lines)

    def _numpy_style_docstring(self, func_name: str, args: List[str], has_return: bool) -> str:
        """Generate NumPy-style docstring"""
        doc_lines = [
            f'"""',
            f'{self._generate_summary(func_name)}',
            ''
        ]

        if args:
            doc_lines.append('Parameters')
            doc_lines.append('----------')
            for arg in args:
                doc_lines.append(f'{arg} : type')
                doc_lines.append(f'    Description of {arg}')
            doc_lines.append('')

        if has_return:
            doc_lines.append('Returns')
            doc_lines.append('-------')
            doc_lines.append('type')
            doc_lines.append('    Description of return value')
            doc_lines.append('')

        doc_lines.append('"""')

        return '\n    '.join(doc_lines)

    def _sphinx_style_docstring(self, func_name: str, args: List[str], has_return: bool) -> str:
        """Generate Sphinx-style docstring"""
        doc_lines = [
            f'"""',
            f'{self._generate_summary(func_name)}',
            ''
        ]

        for arg in args:
            doc_lines.append(f':param {arg}: Description of {arg}')

        if has_return:
            doc_lines.append(':return: Description of return value')

        doc_lines.append('"""')

        return '\n    '.join(doc_lines)

    @staticmethod
    def _generate_summary(func_name: str) -> str:
        """Generate function summary from name"""
        # Convert snake_case to words
        words = func_name.replace('_', ' ').split()

        # Capitalize first word
        if words:
            words[0] = words[0].capitalize()

        return ' '.join(words)

    def generate_class_docstring(self, class_node: ast.ClassDef) -> str:
        """Generate docstring for a class"""
        # Get attributes from __init__
        init_method = None
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                init_method = node
                break

        attributes = []
        if init_method:
            for node in ast.walk(init_method):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                attributes.append(target.attr)

        doc_lines = [
            f'"""',
            f'{class_node.name} class',
            ''
        ]

        if attributes:
            doc_lines.append('Attributes:')
            for attr in attributes:
                doc_lines.append(f'    {attr}: Description of {attr}')
            doc_lines.append('')

        doc_lines.append('"""')

        return '\n    '.join(doc_lines)

    def generate_all(self, code: str) -> List[GeneratedDoc]:
        """
        Generate docstrings for all functions/classes without them

        Args:
            code: Python source code

        Returns:
            List of generated documentation
        """
        docs = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                # Generate for functions
                if isinstance(node, ast.FunctionDef):
                    if not ast.get_docstring(node):
                        docstring = self.generate_function_docstring(node, code)
                        docs.append(GeneratedDoc(
                            target=node.name,
                            docstring=docstring,
                            line=node.lineno
                        ))

                # Generate for classes
                elif isinstance(node, ast.ClassDef):
                    if not ast.get_docstring(node):
                        docstring = self.generate_class_docstring(node)
                        docs.append(GeneratedDoc(
                            target=node.name,
                            docstring=docstring,
                            line=node.lineno
                        ))

        except SyntaxError:
            pass

        return docs


class TestGenerator:
    """
    Automatically generate unit tests for functions
    """

    def __init__(self, framework: str = 'pytest'):
        """
        Args:
            framework: Testing framework ('pytest' or 'unittest')
        """
        self.framework = framework

    def generate_pytest_tests(self, func_node: ast.FunctionDef, code: str) -> str:
        """Generate pytest test cases"""
        func_name = func_node.name
        args = [arg.arg for arg in func_node.args.args if arg.arg != 'self']

        # Generate test code
        test_lines = [
            f'def test_{func_name}_basic():',
            f'    """Test basic functionality of {func_name}"""',
        ]

        # Generate sample arguments
        sample_args = []
        for arg in args:
            # Try to infer type from annotation
            annotation = func_node.args.args[len(sample_args)].annotation
            if annotation:
                if isinstance(annotation, ast.Name):
                    type_name = annotation.id
                    if type_name == 'str':
                        sample_args.append(f'"{arg}_value"')
                    elif type_name == 'int':
                        sample_args.append('42')
                    elif type_name == 'float':
                        sample_args.append('3.14')
                    elif type_name == 'bool':
                        sample_args.append('True')
                    elif type_name == 'list':
                        sample_args.append('[]')
                    elif type_name == 'dict':
                        sample_args.append('{}')
                    else:
                        sample_args.append('None')
                else:
                    sample_args.append('None')
            else:
                # Default to string for unknown types
                sample_args.append(f'"{arg}_value"')

        # Generate function call
        args_str = ', '.join(sample_args)
        test_lines.append(f'    result = {func_name}({args_str})')
        test_lines.append(f'    assert result is not None  # TODO: Add specific assertion')
        test_lines.append('')

        # Generate edge case test
        test_lines.extend([
            f'def test_{func_name}_edge_cases():',
            f'    """Test edge cases for {func_name}"""',
            f'    # TODO: Add edge case tests',
            f'    pass',
            '',
        ])

        # Generate error case test
        test_lines.extend([
            f'def test_{func_name}_errors():',
            f'    """Test error handling in {func_name}"""',
            f'    # TODO: Add error case tests',
            f'    pass',
            '',
        ])

        return '\n'.join(test_lines)

    def generate_unittest_tests(self, func_node: ast.FunctionDef, code: str) -> str:
        """Generate unittest test cases"""
        func_name = func_node.name
        class_name = f'Test{func_name.title().replace("_", "")}'

        test_lines = [
            f'class {class_name}(unittest.TestCase):',
            f'    """Test cases for {func_name}"""',
            '',
            f'    def test_{func_name}_basic(self):',
            f'        """Test basic functionality"""',
            f'        # TODO: Implement test',
            f'        pass',
            '',
            f'    def test_{func_name}_edge_cases(self):',
            f'        """Test edge cases"""',
            f'        # TODO: Implement test',
            f'        pass',
            '',
            f'    def test_{func_name}_errors(self):',
            f'        """Test error handling"""',
            f'        # TODO: Implement test',
            f'        pass',
            '',
        ]

        return '\n'.join(test_lines)

    def generate_all(self, code: str, module_name: str = 'module') -> str:
        """
        Generate complete test file for module

        Args:
            code: Python source code
            module_name: Name of module being tested

        Returns:
            Complete test file content
        """
        test_lines = []

        # Add imports
        if self.framework == 'pytest':
            test_lines.extend([
                f'"""Tests for {module_name}"""',
                'import pytest',
                f'from {module_name} import *',
                '',
                '',
            ])
        else:  # unittest
            test_lines.extend([
                f'"""Tests for {module_name}"""',
                'import unittest',
                f'from {module_name} import *',
                '',
                '',
            ])

        # Parse code and generate tests
        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip private functions
                    if not node.name.startswith('_'):
                        if self.framework == 'pytest':
                            test_code = self.generate_pytest_tests(node, code)
                        else:
                            test_code = self.generate_unittest_tests(node, code)

                        test_lines.append(test_code)

        except SyntaxError:
            pass

        # Add unittest main
        if self.framework == 'unittest':
            test_lines.extend([
                '',
                'if __name__ == "__main__":',
                '    unittest.main()',
            ])

        return '\n'.join(test_lines)


if __name__ == '__main__':
    # Example usage
    test_code = """
def calculate_total(items, tax_rate):
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax

class ShoppingCart:
    def __init__(self):
        self.items = []
        self.total = 0.0

    def add_item(self, item):
        self.items.append(item)
"""

    print("=" * 70)
    print("Code Generators Demo")
    print("=" * 70)

    # Generate documentation
    doc_gen = DocumentationGenerator(style='google')
    docs = doc_gen.generate_all(test_code)

    print("\n📚 Generated Documentation:")
    for doc in docs:
        print(f"\n  Function/Class: {doc.target} (line {doc.line})")
        print(f"  Docstring:")
        for line in doc.docstring.split('\n'):
            print(f"    {line}")

    # Generate tests
    test_gen = TestGenerator(framework='pytest')
    tests = test_gen.generate_all(test_code, module_name='shopping')

    print("\n🧪 Generated Tests (pytest):")
    print(tests)

    print("\n" + "=" * 70)
