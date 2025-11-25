"""
Predictive Code Completion (Phase 2.4)

Goes beyond simple autocomplete to predict what you're trying to implement
based on context, intent, and patterns.

Features:
- Intent detection from context
- Multi-line completions (entire functions/classes)
- Error-aware suggestions (fix common mistakes before they happen)
- Test-driven suggestions (generate implementation from tests)
- Documentation-driven (implement from docstring)
- Pattern-based completions (common algorithms and data structures)

Examples:
    >>> predictor = PredictiveCompletion()
    >>>
    >>> # From docstring
    >>> code = '''
    ... def calculate_fibonacci(n: int) -> int:
    ...     """Calculate nth Fibonacci number using memoization"""
    ... '''
    >>> completion = predictor.complete_from_docstring(code)
    >>>
    >>> # From test
    >>> test = 'assert binary_search([1,2,3,4,5], 3) == 2'
    >>> completion = predictor.complete_from_test(test)
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class CompletionIntent(Enum):
    """Types of completion intents"""
    FUNCTION_BODY = "function_body"           # Complete function implementation
    CLASS_METHODS = "class_methods"           # Add methods to class
    ERROR_HANDLING = "error_handling"         # Add error handling
    TYPE_ANNOTATIONS = "type_annotations"     # Add type hints
    DOCUMENTATION = "documentation"           # Add docstrings
    TESTS = "tests"                          # Generate test cases
    ALGORITHM = "algorithm"                  # Implement common algorithm


@dataclass
class CompletionContext:
    """Context for code completion"""
    intent: CompletionIntent
    cursor_position: int
    surrounding_code: str
    function_signature: Optional[str] = None
    docstring: Optional[str] = None
    test_cases: List[str] = field(default_factory=list)
    error_context: Optional[str] = None
    confidence: float = 1.0


@dataclass
class Completion:
    """A code completion suggestion"""
    code: str
    description: str
    intent: CompletionIntent
    confidence: float
    explanation: str
    metadata: Dict = field(default_factory=dict)


class DocstringParser:
    """Parse docstrings to extract implementation intent"""

    ALGORITHM_KEYWORDS = {
        'binary search': 'binary_search',
        'fibonacci': 'fibonacci',
        'factorial': 'factorial',
        'merge sort': 'merge_sort',
        'quick sort': 'quick_sort',
        'depth-first': 'dfs',
        'breadth-first': 'bfs',
        'dynamic programming': 'dynamic_programming',
        'memoization': 'memoization',
        'recursion': 'recursion',
    }

    def parse(self, docstring: str) -> Dict:
        """Parse docstring to extract implementation hints"""

        hints = {
            'algorithm': None,
            'complexity': None,
            'parameters': [],
            'returns': None,
            'constraints': []
        }

        if not docstring:
            return hints

        doc_lower = docstring.lower()

        # Detect algorithm
        for keyword, algo_type in self.ALGORITHM_KEYWORDS.items():
            if keyword in doc_lower:
                hints['algorithm'] = algo_type
                break

        # Extract complexity hints
        complexity_match = re.search(r'O\(([^)]+)\)', docstring)
        if complexity_match:
            hints['complexity'] = complexity_match.group(0)

        # Extract parameter descriptions
        param_pattern = r'(?:Args?|Parameters?):\s*\n\s*(\w+).*?:(.+?)(?=\n\s*\w+:|\n\n|$)'
        for match in re.finditer(param_pattern, docstring, re.MULTILINE | re.DOTALL):
            hints['parameters'].append({
                'name': match.group(1),
                'description': match.group(2).strip()
            })

        # Extract return description
        return_match = re.search(r'Returns?:\s*(.+?)(?=\n\n|$)', docstring, re.DOTALL)
        if return_match:
            hints['returns'] = return_match.group(1).strip()

        return hints


class AlgorithmTemplates:
    """Templates for common algorithms"""

    @staticmethod
    def binary_search(func_name: str, array_param: str, target_param: str) -> str:
        return f'''def {func_name}({array_param}: list, {target_param}) -> int:
    """Binary search implementation"""
    left, right = 0, len({array_param}) - 1

    while left <= right:
        mid = (left + right) // 2

        if {array_param}[mid] == {target_param}:
            return mid
        elif {array_param}[mid] < {target_param}:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found
'''

    @staticmethod
    def fibonacci_memoized(func_name: str, param: str) -> str:
        return f'''def {func_name}({param}: int) -> int:
    """Calculate Fibonacci number with memoization"""
    cache = {{}}

    def fib(k: int) -> int:
        if k in cache:
            return cache[k]

        if k <= 1:
            return k

        cache[k] = fib(k - 1) + fib(k - 2)
        return cache[k]

    return fib({param})
'''

    @staticmethod
    def factorial(func_name: str, param: str) -> str:
        return f'''def {func_name}({param}: int) -> int:
    """Calculate factorial"""
    if {param} <= 1:
        return 1
    return {param} * {func_name}({param} - 1)
'''

    @staticmethod
    def merge_sort(func_name: str, array_param: str) -> str:
        return f'''def {func_name}({array_param}: list) -> list:
    """Merge sort implementation"""
    if len({array_param}) <= 1:
        return {array_param}

    mid = len({array_param}) // 2
    left = {func_name}({array_param}[:mid])
    right = {func_name}({array_param}[mid:])

    return merge(left, right)

def merge(left: list, right: list) -> list:
    """Merge two sorted lists"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''

    @staticmethod
    def dfs_graph(func_name: str, graph_param: str, start_param: str) -> str:
        return f'''def {func_name}({graph_param}: dict, {start_param}) -> list:
    """Depth-first search traversal"""
    visited = set()
    result = []

    def dfs(node):
        if node in visited:
            return

        visited.add(node)
        result.append(node)

        for neighbor in {graph_param}.get(node, []):
            dfs(neighbor)

    dfs({start_param})
    return result
'''


class TestCaseAnalyzer:
    """Analyze test cases to infer implementation"""

    def analyze_assertion(self, assertion: str) -> Dict:
        """Analyze test assertion to extract requirements"""

        info = {
            'function_name': None,
            'inputs': [],
            'expected_output': None,
            'edge_case': False
        }

        # Parse assertion: assert func(args) == expected
        match = re.match(r'assert\s+(\w+)\((.*?)\)\s*==\s*(.+)', assertion.strip())
        if match:
            info['function_name'] = match.group(1)

            # Parse arguments
            args_str = match.group(2)
            info['inputs'] = [arg.strip() for arg in args_str.split(',') if arg.strip()]

            info['expected_output'] = match.group(3).strip()

            # Detect edge cases
            if any(keyword in args_str.lower() for keyword in ['[]', 'none', '0', 'empty']):
                info['edge_case'] = True

        return info

    def infer_implementation_from_tests(self, test_cases: List[str]) -> Dict:
        """Infer implementation requirements from test cases"""

        requirements = {
            'function_name': None,
            'input_types': [],
            'output_type': None,
            'edge_cases': [],
            'normal_cases': []
        }

        for test in test_cases:
            info = self.analyze_assertion(test)

            if info['function_name']:
                requirements['function_name'] = info['function_name']

                if info['edge_case']:
                    requirements['edge_cases'].append(info)
                else:
                    requirements['normal_cases'].append(info)

        return requirements


class ErrorAwareCompletion:
    """Provide completions to prevent common errors"""

    COMMON_ERRORS = {
        'division_by_zero': {
            'pattern': r'/\s*(\w+)',
            'completion': lambda var: f'''if {var} == 0:
    raise ValueError("Division by zero")
result = ... / {var}
'''
        },
        'null_reference': {
            'pattern': r'\.(\w+)\(',
            'completion': lambda obj: f'''if {obj} is None:
    raise ValueError("Object is None")
{obj}.method()
'''
        },
        'index_out_of_bounds': {
            'pattern': r'\[(\w+)\]',
            'completion': lambda var: f'''if {var} >= len(collection):
    raise IndexError("Index out of bounds")
value = collection[{var}]
'''
        }
    }

    def suggest_error_prevention(self, code: str) -> List[Completion]:
        """Suggest error prevention code"""

        suggestions = []

        for error_type, config in self.COMMON_ERRORS.items():
            pattern = config['pattern']
            matches = re.finditer(pattern, code)

            for match in matches:
                var = match.group(1)
                completion_code = config['completion'](var)

                suggestions.append(Completion(
                    code=completion_code,
                    description=f"Prevent {error_type.replace('_', ' ')}",
                    intent=CompletionIntent.ERROR_HANDLING,
                    confidence=0.7,
                    explanation=f"Add check to prevent {error_type}",
                    metadata={'error_type': error_type, 'variable': var}
                ))

        return suggestions


class PredictiveCompletion:
    """Main predictive completion engine"""

    def __init__(self):
        self.docstring_parser = DocstringParser()
        self.test_analyzer = TestCaseAnalyzer()
        self.error_completion = ErrorAwareCompletion()
        self.templates = AlgorithmTemplates()

    def complete_from_docstring(self, code: str) -> List[Completion]:
        """Generate completion from function docstring"""

        completions = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    if not docstring:
                        continue

                    # Parse docstring for hints
                    hints = self.docstring_parser.parse(docstring)

                    # Generate completion based on algorithm hint
                    if hints['algorithm']:
                        completion = self._generate_algorithm_completion(
                            node.name,
                            hints['algorithm'],
                            node.args
                        )
                        if completion:
                            completions.append(completion)

        except SyntaxError:
            pass

        return completions

    def _generate_algorithm_completion(self, func_name: str, algorithm: str, args: ast.arguments) -> Optional[Completion]:
        """Generate algorithm implementation"""

        # Extract parameter names
        param_names = [arg.arg for arg in args.args]

        template_method = getattr(self.templates, algorithm, None)
        if not template_method:
            return None

        # Generate code based on template
        if algorithm == 'binary_search':
            if len(param_names) >= 2:
                code = template_method(func_name, param_names[0], param_names[1])
            else:
                code = template_method(func_name, 'arr', 'target')
        elif algorithm == 'fibonacci' or algorithm == 'factorial':
            if param_names:
                code = template_method(func_name, param_names[0])
            else:
                code = template_method(func_name, 'n')
        elif algorithm == 'merge_sort':
            if param_names:
                code = template_method(func_name, param_names[0])
            else:
                code = template_method(func_name, 'arr')
        elif algorithm == 'dfs':
            if len(param_names) >= 2:
                code = template_method(func_name, param_names[0], param_names[1])
            else:
                code = template_method(func_name, 'graph', 'start')
        else:
            return None

        return Completion(
            code=code,
            description=f"Implement {algorithm.replace('_', ' ')} algorithm",
            intent=CompletionIntent.ALGORITHM,
            confidence=0.9,
            explanation=f"Generated {algorithm} implementation from docstring",
            metadata={'algorithm': algorithm}
        )

    def complete_from_test(self, test_code: str) -> List[Completion]:
        """Generate implementation from test cases"""

        completions = []

        # Extract test assertions
        assertions = re.findall(r'assert\s+.+', test_code)

        if not assertions:
            return completions

        # Analyze test requirements
        requirements = self.test_analyzer.infer_implementation_from_tests(assertions)

        if requirements['function_name']:
            # Generate basic implementation structure
            func_name = requirements['function_name']

            # Infer function signature from test inputs
            if requirements['normal_cases']:
                first_case = requirements['normal_cases'][0]
                params = ', '.join(f'arg{i}' for i in range(len(first_case['inputs'])))

                impl = f'''def {func_name}({params}):
    """
    Implementation generated from test cases.

    Test cases:
'''

                for case in requirements['normal_cases'][:3]:
                    impl += f"    - {func_name}({', '.join(case['inputs'])}) == {case['expected_output']}\n"

                impl += '''    """
    # TODO: Implement based on test cases
    pass
'''

                completions.append(Completion(
                    code=impl,
                    description=f"Implement {func_name} from tests",
                    intent=CompletionIntent.TESTS,
                    confidence=0.75,
                    explanation="Generated function skeleton from test cases",
                    metadata={'test_count': len(requirements['normal_cases'])}
                ))

        return completions

    def complete_function_body(self, code: str, function_name: str) -> List[Completion]:
        """Complete function body from signature and context"""

        completions = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Check if function is empty
                    if len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Expr)):
                        docstring = ast.get_docstring(node)

                        # Try docstring-driven completion
                        if docstring:
                            doc_completions = self.complete_from_docstring(code)
                            completions.extend(doc_completions)

                        # Generate basic skeleton
                        params = ', '.join(arg.arg for arg in node.args.args)
                        skeleton = f'''def {function_name}({params}):
    """Function implementation"""
    # Validate inputs
    if not {node.args.args[0].arg if node.args.args else 'input'}:
        raise ValueError("Invalid input")

    # Process
    result = None  # TODO: Implement logic

    # Return
    return result
'''

                        completions.append(Completion(
                            code=skeleton,
                            description="Basic function skeleton with validation",
                            intent=CompletionIntent.FUNCTION_BODY,
                            confidence=0.6,
                            explanation="Generated basic structure with input validation",
                            metadata={'params': params}
                        ))

        except SyntaxError:
            pass

        return completions

    def add_type_annotations(self, code: str) -> List[Completion]:
        """Suggest type annotations"""

        completions = []

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if missing type annotations
                    has_annotations = any(arg.annotation for arg in node.args.args)
                    has_return = node.returns is not None

                    if not has_annotations or not has_return:
                        # Generate annotated version
                        params_with_types = []
                        for arg in node.args.args:
                            if arg.annotation:
                                params_with_types.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                            else:
                                # Infer type from name
                                inferred_type = self._infer_type_from_name(arg.arg)
                                params_with_types.append(f"{arg.arg}: {inferred_type}")

                        params_str = ', '.join(params_with_types)
                        return_type = ast.unparse(node.returns) if node.returns else "Any"

                        annotated_sig = f"def {node.name}({params_str}) -> {return_type}:"

                        completions.append(Completion(
                            code=annotated_sig,
                            description=f"Add type annotations to {node.name}",
                            intent=CompletionIntent.TYPE_ANNOTATIONS,
                            confidence=0.7,
                            explanation="Added type hints based on naming conventions",
                            metadata={'function': node.name}
                        ))

        except SyntaxError:
            pass

        return completions

    def _infer_type_from_name(self, name: str) -> str:
        """Infer type from parameter name"""

        type_hints = {
            'count': 'int',
            'size': 'int',
            'length': 'int',
            'index': 'int',
            'num': 'int',
            'id': 'int',
            'name': 'str',
            'text': 'str',
            'message': 'str',
            'email': 'str',
            'path': 'str',
            'filename': 'str',
            'enabled': 'bool',
            'active': 'bool',
            'valid': 'bool',
            'items': 'List',
            'data': 'Dict',
            'config': 'Dict',
            'options': 'Dict',
        }

        name_lower = name.lower()

        for keyword, type_hint in type_hints.items():
            if keyword in name_lower:
                return type_hint

        return 'Any'

    def suggest_error_handling(self, code: str) -> List[Completion]:
        """Suggest error handling improvements"""
        return self.error_completion.suggest_error_prevention(code)

    def format_completion(self, completion: Completion) -> str:
        """Format completion for display"""

        lines = []
        lines.append("=" * 60)
        lines.append(f"COMPLETION: {completion.description}")
        lines.append("=" * 60)
        lines.append(f"Intent: {completion.intent.value}")
        lines.append(f"Confidence: {completion.confidence:.0%}")
        lines.append(f"Explanation: {completion.explanation}")
        lines.append("")
        lines.append("Code:")
        lines.append("-" * 60)
        lines.append(completion.code)
        lines.append("-" * 60)

        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    predictor = PredictiveCompletion()

    # Test 1: Completion from docstring
    print("TEST 1: Completion from docstring")
    print("=" * 80)

    code1 = '''
def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number using memoization"""
    pass
'''

    completions = predictor.complete_from_docstring(code1)
    for comp in completions:
        print(predictor.format_completion(comp))

    print("\n\n")

    # Test 2: Completion from test
    print("TEST 2: Completion from test")
    print("=" * 80)

    test_code = '''
def test_binary_search():
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3, 4, 5], 6) == -1
    assert binary_search([], 1) == -1
'''

    completions = predictor.complete_from_test(test_code)
    for comp in completions:
        print(predictor.format_completion(comp))

    print("\n\n")

    # Test 3: Type annotations
    print("TEST 3: Type annotation suggestions")
    print("=" * 80)

    code3 = '''
def process_data(items, count, enabled):
    """Process some data"""
    pass
'''

    completions = predictor.add_type_annotations(code3)
    for comp in completions:
        print(predictor.format_completion(comp))
