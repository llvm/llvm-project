#!/usr/bin/env python3
"""
Code Formatting and Style Validation
Auto-format code and validate against style guidelines

Formatters: black, autopep8, yapf
Validators: pylint, flake8, mypy
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class FormatterType(Enum):
    """Supported code formatters"""
    BLACK = "black"
    AUTOPEP8 = "autopep8"
    YAPF = "yapf"


class ValidatorType(Enum):
    """Supported style validators"""
    PYLINT = "pylint"
    FLAKE8 = "flake8"
    MYPY = "mypy"


@dataclass
class FormattingResult:
    """Result from code formatting"""
    formatted_code: str
    changed: bool
    formatter: FormatterType
    error: Optional[str] = None


@dataclass
class ValidationResult:
    """Result from style validation"""
    score: float
    issues: List[Dict]
    passes: bool
    validator: ValidatorType
    error: Optional[str] = None


class CodeFormatter:
    """
    Auto-format code with multiple formatter backends

    Example:
        formatter = CodeFormatter()

        # Format with black
        result = formatter.format_code(code, formatter='black')

        # Validate style
        validation = formatter.validate_style(code)
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: Print formatting details
        """
        self.verbose = verbose

    def format_code(self, code: str,
                    formatter: str = 'black',
                    line_length: int = 88) -> FormattingResult:
        """
        Format code with specified formatter

        Args:
            code: Python source code
            formatter: Formatter to use ('black', 'autopep8', 'yapf')
            line_length: Max line length (default: 88 for black)

        Returns:
            FormattingResult
        """
        formatter_type = FormatterType(formatter)

        try:
            if formatter_type == FormatterType.BLACK:
                formatted = self._format_with_black(code, line_length)
            elif formatter_type == FormatterType.AUTOPEP8:
                formatted = self._format_with_autopep8(code, line_length)
            elif formatter_type == FormatterType.YAPF:
                formatted = self._format_with_yapf(code, line_length)
            else:
                return FormattingResult(
                    formatted_code=code,
                    changed=False,
                    formatter=formatter_type,
                    error=f"Unsupported formatter: {formatter}"
                )

            changed = formatted != code

            if self.verbose and changed:
                print(f"✨ Formatted with {formatter} (changed: {changed})")

            return FormattingResult(
                formatted_code=formatted,
                changed=changed,
                formatter=formatter_type,
                error=None
            )

        except Exception as e:
            return FormattingResult(
                formatted_code=code,
                changed=False,
                formatter=formatter_type,
                error=str(e)
            )

    def _format_with_black(self, code: str, line_length: int) -> str:
        """Format with black"""
        try:
            import black

            mode = black.Mode(line_length=line_length)
            return black.format_str(code, mode=mode)

        except ImportError:
            # Fallback to command-line black
            return self._format_with_cli('black', code, [f'--line-length={line_length}', '-'])

    def _format_with_autopep8(self, code: str, line_length: int) -> str:
        """Format with autopep8"""
        try:
            import autopep8

            return autopep8.fix_code(code, options={'max_line_length': line_length})

        except ImportError:
            return self._format_with_cli('autopep8', code, [f'--max-line-length={line_length}', '-'])

    def _format_with_yapf(self, code: str, line_length: int) -> str:
        """Format with yapf"""
        try:
            from yapf.yapflib.yapf_api import FormatCode

            style = {'based_on_style': 'pep8', 'column_limit': line_length}
            formatted, changed = FormatCode(code, style_config=style)
            return formatted

        except ImportError:
            return self._format_with_cli('yapf', code, [f'--style={{column_limit:{line_length}}}'])

    def _format_with_cli(self, tool: str, code: str, args: List[str]) -> str:
        """Fallback: format using command-line tool"""
        try:
            result = subprocess.run(
                [tool] + args,
                input=code,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                return result.stdout
            else:
                raise RuntimeError(f"{tool} failed: {result.stderr}")

        except FileNotFoundError:
            raise ImportError(f"{tool} not installed. Install with: pip install {tool}")

    def validate_style(self, code: str,
                       validators: Optional[List[str]] = None,
                       min_score: float = 8.0) -> Dict[ValidatorType, ValidationResult]:
        """
        Validate code against style guidelines

        Args:
            code: Python source code
            validators: List of validators ('pylint', 'flake8', 'mypy')
            min_score: Minimum passing score (default: 8.0/10 for pylint)

        Returns:
            Dict of validation results
        """
        if validators is None:
            validators = ['flake8']  # Default to flake8 (fastest)

        results = {}

        for validator_name in validators:
            validator_type = ValidatorType(validator_name)

            try:
                if validator_type == ValidatorType.PYLINT:
                    result = self._validate_with_pylint(code, min_score)
                elif validator_type == ValidatorType.FLAKE8:
                    result = self._validate_with_flake8(code)
                elif validator_type == ValidatorType.MYPY:
                    result = self._validate_with_mypy(code)
                else:
                    result = ValidationResult(
                        score=0.0,
                        issues=[],
                        passes=False,
                        validator=validator_type,
                        error=f"Unsupported validator: {validator_name}"
                    )

                results[validator_type] = result

                if self.verbose:
                    status = "✓" if result.passes else "✗"
                    print(f"{status} {validator_name}: {result.score:.1f}/10 ({len(result.issues)} issues)")

            except Exception as e:
                results[validator_type] = ValidationResult(
                    score=0.0,
                    issues=[],
                    passes=False,
                    validator=validator_type,
                    error=str(e)
                )

        return results

    def _validate_with_pylint(self, code: str, min_score: float) -> ValidationResult:
        """Validate with pylint"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ['pylint', temp_file, '--output-format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse JSON output
            import json
            issues = json.loads(result.stdout) if result.stdout else []

            # Calculate score (10 - number of issues / 10)
            score = max(0.0, 10.0 - len(issues) / 10.0)

            return ValidationResult(
                score=score,
                issues=issues,
                passes=score >= min_score,
                validator=ValidatorType.PYLINT
            )

        except FileNotFoundError:
            raise ImportError("pylint not installed. Install with: pip install pylint")

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def _validate_with_flake8(self, code: str) -> ValidationResult:
        """Validate with flake8"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ['flake8', temp_file, '--format=json'],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Parse output (flake8 JSON format)
            issues = []
            if result.stdout:
                import json
                try:
                    data = json.loads(result.stdout)
                    issues = data.get(temp_file, [])
                except json.JSONDecodeError:
                    # Fallback: parse line-based output
                    issues = result.stdout.strip().split('\n') if result.stdout.strip() else []

            score = 10.0 if len(issues) == 0 else max(0.0, 10.0 - len(issues))

            return ValidationResult(
                score=score,
                issues=issues,
                passes=len(issues) == 0,
                validator=ValidatorType.FLAKE8
            )

        except FileNotFoundError:
            raise ImportError("flake8 not installed. Install with: pip install flake8")

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def _validate_with_mypy(self, code: str) -> ValidationResult:
        """Validate with mypy (type checking)"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ['mypy', temp_file, '--show-error-codes'],
                capture_output=True,
                text=True,
                timeout=30
            )

            issues = result.stdout.strip().split('\n') if result.stdout.strip() else []
            score = 10.0 if len(issues) == 0 else max(0.0, 10.0 - len(issues))

            return ValidationResult(
                score=score,
                issues=issues,
                passes=len(issues) == 0,
                validator=ValidatorType.MYPY
            )

        except FileNotFoundError:
            raise ImportError("mypy not installed. Install with: pip install mypy")

        finally:
            Path(temp_file).unlink(missing_ok=True)

    def format_and_validate(self, code: str,
                           formatter: str = 'black',
                           validators: Optional[List[str]] = None) -> Tuple[FormattingResult, Dict]:
        """
        Format code and then validate

        Args:
            code: Python source code
            formatter: Formatter to use
            validators: Validators to run

        Returns:
            (FormattingResult, validation_results)
        """
        # Format first
        format_result = self.format_code(code, formatter=formatter)

        # Validate formatted code
        validation_results = self.validate_style(
            format_result.formatted_code,
            validators=validators
        )

        return format_result, validation_results


def main():
    """Test code formatter"""
    test_code = """
def badly_formatted_function(x,y,z):
    result=x+y+z
    if result>10:
        print( "Large result" )
    return result

class   MyClass:
    def __init__(  self,  value  ):
        self.value=value
"""

    print("="*70)
    print("Code Formatting Demo")
    print("="*70)

    formatter = CodeFormatter(verbose=True)

    # Test black
    print("\n🎨 Formatting with black:")
    result = formatter.format_code(test_code, formatter='black')
    print(result.formatted_code)

    # Test validation
    print("\n📋 Validating formatted code:")
    validation = formatter.validate_style(result.formatted_code, validators=['flake8'])

    for validator_type, val_result in validation.items():
        print(f"\n{validator_type.value}:")
        print(f"  Score: {val_result.score:.1f}/10")
        print(f"  Passes: {val_result.passes}")
        print(f"  Issues: {len(val_result.issues)}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
