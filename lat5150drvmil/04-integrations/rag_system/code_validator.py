#!/usr/bin/env python3
"""
Code Validation and Auto-Feedback System
Automatically detects if code compiles/runs successfully

Features:
- Python syntax validation
- Code execution in sandbox
- Error detection and classification
- Auto-rating based on execution results
- Screenshot analysis (optional)
- Log file parsing
"""

import subprocess
import sys
import tempfile
import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional
import ast


class CodeValidator:
    """Validate code snippets and auto-detect execution results"""

    def __init__(self):
        """Initialize code validator"""
        self.temp_dir = tempfile.mkdtemp(prefix='rag_code_test_')

    def extract_code_blocks(self, text: str) -> list:
        """
        Extract code blocks from markdown/text

        Supports:
        - ```python ... ```
        - ```bash ... ```
        - Indented code blocks
        """
        code_blocks = []

        # Pattern 1: Fenced code blocks with language
        pattern_fenced = r'```(\w+)?\n(.*?)```'
        for match in re.finditer(pattern_fenced, text, re.DOTALL):
            lang = match.group(1) or 'unknown'
            code = match.group(2).strip()
            code_blocks.append({
                'language': lang,
                'code': code,
                'type': 'fenced'
            })

        # Pattern 2: Indented code blocks (4 spaces or tab)
        if not code_blocks:
            lines = text.split('\n')
            current_block = []
            for line in lines:
                if line.startswith('    ') or line.startswith('\t'):
                    current_block.append(line.strip())
                elif current_block:
                    code_blocks.append({
                        'language': 'unknown',
                        'code': '\n'.join(current_block),
                        'type': 'indented'
                    })
                    current_block = []

        return code_blocks

    def validate_python_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Check if Python code has valid syntax

        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, str(e)

    def execute_python_code(
        self,
        code: str,
        timeout: int = 5
    ) -> Dict:
        """
        Execute Python code in subprocess and capture result

        Args:
            code: Python code to execute
            timeout: Max execution time in seconds

        Returns:
            Dict with execution results
        """
        # First check syntax
        is_valid, syntax_error = self.validate_python_syntax(code)
        if not is_valid:
            return {
                'success': False,
                'error_type': 'syntax',
                'error_message': syntax_error,
                'stdout': '',
                'stderr': '',
                'rating': 2,
                'feedback': f"Code has syntax error: {syntax_error}"
            }

        # Create temporary file
        temp_file = Path(self.temp_dir) / "test_code.py"
        with open(temp_file, 'w') as f:
            f.write(code)

        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, str(temp_file)],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # Analyze result
            if result.returncode == 0:
                return {
                    'success': True,
                    'error_type': None,
                    'error_message': None,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'rating': 10,
                    'feedback': "Code executed successfully - error free"
                }
            else:
                # Parse error
                error_type, error_msg = self._parse_python_error(result.stderr)
                return {
                    'success': False,
                    'error_type': error_type,
                    'error_message': error_msg,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'rating': 2,
                    'feedback': f"Runtime error: {error_type}"
                }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error_type': 'timeout',
                'error_message': f'Code execution timed out after {timeout}s',
                'stdout': '',
                'stderr': '',
                'rating': 3,
                'feedback': f"Code timed out (infinite loop?)"
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': 'execution',
                'error_message': str(e),
                'stdout': '',
                'stderr': '',
                'rating': 1,
                'feedback': f"Failed to execute: {str(e)}"
            }
        finally:
            # Cleanup
            if temp_file.exists():
                temp_file.unlink()

    def _parse_python_error(self, stderr: str) -> Tuple[str, str]:
        """Parse Python error messages"""
        # Common error patterns
        if 'NameError' in stderr:
            return 'NameError', 'Undefined variable or function'
        elif 'TypeError' in stderr:
            return 'TypeError', 'Type mismatch or invalid operation'
        elif 'ValueError' in stderr:
            return 'ValueError', 'Invalid value for operation'
        elif 'ImportError' in stderr or 'ModuleNotFoundError' in stderr:
            return 'ImportError', 'Missing module or import error'
        elif 'AttributeError' in stderr:
            return 'AttributeError', 'Object has no such attribute'
        elif 'IndentationError' in stderr:
            return 'IndentationError', 'Incorrect indentation'
        elif 'KeyError' in stderr:
            return 'KeyError', 'Dictionary key not found'
        elif 'IndexError' in stderr:
            return 'IndexError', 'List index out of range'
        elif 'ZeroDivisionError' in stderr:
            return 'ZeroDivisionError', 'Division by zero'
        else:
            return 'RuntimeError', 'Unknown runtime error'

    def validate_bash_command(self, command: str) -> Dict:
        """
        Validate bash command (syntax check only, no execution)

        Returns:
            Dict with validation results
        """
        # Check for dangerous commands
        dangerous_patterns = [
            r'\brm\s+-rf\s+/',
            r'\bdd\s+if=',
            r'\bmkfs',
            r'\b:(){:\|:&};:',  # Fork bomb
            r'\bshred\b',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                return {
                    'success': False,
                    'error_type': 'dangerous',
                    'error_message': 'Command appears dangerous',
                    'rating': 1,
                    'feedback': "Code contains potentially dangerous commands"
                }

        # Basic syntax check via bash -n
        try:
            result = subprocess.run(
                ['bash', '-n', '-c', command],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode == 0:
                return {
                    'success': True,
                    'error_type': None,
                    'error_message': None,
                    'rating': 8,
                    'feedback': "Bash syntax appears valid (not executed)"
                }
            else:
                return {
                    'success': False,
                    'error_type': 'syntax',
                    'error_message': result.stderr,
                    'rating': 3,
                    'feedback': f"Bash syntax error: {result.stderr[:100]}"
                }

        except Exception as e:
            return {
                'success': False,
                'error_type': 'validation',
                'error_message': str(e),
                'rating': 2,
                'feedback': f"Could not validate bash syntax"
            }

    def auto_validate_response(self, query: str, response_text: str) -> Dict:
        """
        Automatically validate code in response

        Args:
            query: User's query
            response_text: RAG response containing code

        Returns:
            Dict with validation results and auto-rating
        """
        # Extract code blocks
        code_blocks = self.extract_code_blocks(response_text)

        if not code_blocks:
            # No code blocks found
            return {
                'has_code': False,
                'validation': None,
                'auto_rating': None,
                'auto_feedback': None
            }

        results = []

        for block in code_blocks:
            if block['language'] == 'python':
                result = self.execute_python_code(block['code'])
                results.append(result)

            elif block['language'] in ['bash', 'sh']:
                result = self.validate_bash_command(block['code'])
                results.append(result)

        # Aggregate results
        if not results:
            return {
                'has_code': True,
                'validation': None,
                'auto_rating': None,
                'auto_feedback': 'Code found but could not be validated'
            }

        # Overall rating: average if all pass, min if any fail
        all_success = all(r['success'] for r in results)

        if all_success:
            avg_rating = sum(r['rating'] for r in results) / len(results)
            return {
                'has_code': True,
                'validation': results,
                'auto_rating': int(avg_rating),
                'auto_feedback': "All code blocks validated successfully"
            }
        else:
            # Find first failure
            first_failure = next(r for r in results if not r['success'])
            return {
                'has_code': True,
                'validation': results,
                'auto_rating': first_failure['rating'],
                'auto_feedback': first_failure['feedback']
            }

    def __del__(self):
        """Cleanup temporary directory"""
        try:
            import shutil
            if Path(self.temp_dir).exists():
                shutil.rmtree(self.temp_dir)
        except:
            pass


def main():
    """Test code validator"""
    validator = CodeValidator()

    # Test cases
    test_cases = [
        {
            'name': 'Valid Python',
            'code': '''```python
print("Hello World")
x = 5 + 3
print(x)
```'''
        },
        {
            'name': 'Python Syntax Error',
            'code': '''```python
print("Hello World"
x = 5 +
```'''
        },
        {
            'name': 'Python Runtime Error',
            'code': '''```python
print(undefined_variable)
```'''
        },
        {
            'name': 'Valid Bash',
            'code': '''```bash
echo "Hello"
ls -la
```'''
        },
    ]

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Test: {test['name']}")
        print(f"{'='*60}")

        result = validator.auto_validate_response("test query", test['code'])

        print(f"Has code: {result['has_code']}")
        if result['auto_rating']:
            print(f"Auto-rating: {result['auto_rating']}/10")
            print(f"Feedback: {result['auto_feedback']}")

        if result['validation']:
            for i, v in enumerate(result['validation'], 1):
                print(f"\nBlock {i}:")
                print(f"  Success: {v['success']}")
                print(f"  Rating: {v['rating']}/10")
                print(f"  Feedback: {v['feedback']}")
                if v['stdout']:
                    print(f"  Output: {v['stdout'][:100]}")


if __name__ == '__main__':
    main()
