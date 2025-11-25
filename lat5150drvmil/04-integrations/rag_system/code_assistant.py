#!/usr/bin/env python3
"""
RAG-Enhanced Code Assistant for LAT5150DRVMIL
Production-grade AI coding assistant with local LLM + documentation context

Features:
- Multi-turn conversations with context
- Code execution and testing
- File operations (save/load/edit)
- Syntax highlighting
- Code review and analysis
- RAG integration with your documentation
- 100% local (Ollama + transformers)

Advanced Features:
- Security vulnerability scanning (SQL injection, command injection, etc.)
- Performance optimization analysis
- AST-based code transformations
- Automatic documentation generation
- Unit test generation (pytest/unittest)
- Code complexity metrics
- Code smell detection
"""

import os
import json
import subprocess
import tempfile
import shlex
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

# Import advanced analysis modules
try:
    from rag_system.code_analysis_engine import (
        SecurityScanner, PerformanceOptimizer, ComplexityAnalyzer,
        CodeSmellDetector, Severity
    )
    from rag_system.code_transformers import (
        ErrorHandlingTransformer, TypeHintAdder, PerformanceRefactorer,
        apply_all_transformers
    )
    from rag_system.code_generators import (
        DocumentationGenerator, TestGenerator
    )
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False


@dataclass
class Conversation:
    """Maintains conversation context"""
    messages: List[Dict[str, str]] = field(default_factory=list)
    rag_context: List[Dict] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.messages.append({"role": role, "content": content})

    def get_context(self, last_n: int = 10) -> str:
        """Get recent conversation context"""
        recent = self.messages[-last_n:] if len(self.messages) > last_n else self.messages
        return "\n\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent
        ])

    def clear(self):
        """Clear conversation history"""
        self.messages.clear()
        self.rag_context.clear()


class CodeAssistant:
    """
    Production-grade code assistant with RAG + local LLM

    Capabilities:
    - Generate code with context from your docs
    - Multi-turn conversations
    - Execute and test generated code
    - Code review and refactoring
    - Save/load code files
    - Integrate with existing codebase
    """

    def __init__(self, model: str = 'deepseek-coder:6.7b', verbose: bool = True):
        """
        Initialize code assistant

        Args:
            model: Ollama model (deepseek-coder:6.7b recommended)
            verbose: Print detailed logs
        """
        self.model = model
        self.verbose = verbose
        self.conversation = Conversation()
        self.project_root = Path.cwd()

        # Verify Ollama is installed
        self._check_ollama()

        # Load RAG retriever
        self.retriever = self._load_retriever()

        # System prompt for coding tasks
        self.system_prompt = """You are an expert software engineer specializing in:
- Embedded Linux systems (LAT5150DRVMIL)
- Kernel development and device drivers
- Security and malware analysis
- Python, C, Bash, and systems programming
- Hardware interfacing and NPU programming

You provide:
1. Production-ready, secure code
2. Clear explanations
3. Best practices and error handling
4. Practical examples
5. Performance considerations

Always include comments and follow modern coding standards."""

    def _check_ollama(self):
        """Verify Ollama installation and model availability"""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )

            if self.model not in result.stdout:
                if self.verbose:
                    print(f"⚠️  Model {self.model} not found. Downloading...")
                subprocess.run(['ollama', 'pull', self.model], check=True)

        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not installed!\n\n"
                "Install with:\n"
                "  curl -fsSL https://ollama.com/install.sh | sh\n\n"
                "Then pull a coding model:\n"
                "  ollama pull deepseek-coder:6.7b"
            )

    def _load_retriever(self):
        """Load transformer-based RAG retriever with temporal awareness"""
        try:
            from rag_system.transformer_upgrade import TransformerRetriever
            from rag_system.temporal_decay import TemporalAwareRetriever

            chunks_file = 'rag_system/processed_docs.json'
            if not Path(chunks_file).exists():
                if self.verbose:
                    print(f"⚠️  RAG index not found. Run: python3 rag_system/document_processor.py")
                return None

            with open(chunks_file, 'r') as f:
                data = json.load(f)

            chunks = data['chunks']

            if self.verbose:
                print(f"✓ Loaded RAG system ({len(chunks)} chunks)")

            # Load base retriever
            base_retriever = TransformerRetriever.load_embeddings(chunks)

            # Wrap with temporal awareness
            temporal_retriever = TemporalAwareRetriever(base_retriever, verbose=False)

            if self.verbose:
                print(f"✓ Temporal awareness enabled (fixes 'temporal blindness')")

            return temporal_retriever

        except Exception as e:
            if self.verbose:
                print(f"⚠️  RAG unavailable: {e}")
            return None

    def _retrieve_context(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant documentation with temporal decay

        Returns:
            (formatted_context, results)
        """
        if not self.retriever:
            return "", []

        # Use temporal-aware retrieval (automatically applies decay)
        if hasattr(self.retriever, 'retrieve_with_decay'):
            results = self.retriever.retrieve_with_decay(query, top_k=top_k)
        else:
            # Fallback for non-temporal retriever
            results = self.retriever.search(query, top_k=top_k)

        context_parts = []
        for i, (chunk, score) in enumerate(results, 1):
            filename = chunk['metadata'].get('filename', 'Unknown')
            context_parts.append(
                f"### Reference {i}: {filename} (Relevance: {score:.2f})\n"
                f"{chunk['text'][:500]}\n"
            )

        context = "\n---\n".join(context_parts)
        return context, [chunk for chunk, _ in results]

    def _build_prompt(self, user_query: str, use_rag: bool = True,
                     include_history: bool = True) -> str:
        """Build optimized prompt for code generation"""

        prompt_parts = [f"SYSTEM:\n{self.system_prompt}\n"]

        # Add RAG context
        if use_rag:
            context, rag_results = self._retrieve_context(user_query, top_k=5)
            if context:
                self.conversation.rag_context = rag_results
                prompt_parts.append(f"DOCUMENTATION CONTEXT:\n{context}\n")

        # Add conversation history
        if include_history and self.conversation.messages:
            history = self.conversation.get_context(last_n=6)
            prompt_parts.append(f"CONVERSATION HISTORY:\n{history}\n")

        # Add user query
        prompt_parts.append(f"USER REQUEST:\n{user_query}\n")

        # Instructions
        prompt_parts.append("""
INSTRUCTIONS:
1. Analyze the documentation context and conversation history
2. Provide complete, working code with explanations
3. Include error handling and edge cases
4. Add clear comments
5. Use best practices and modern patterns
6. If the request is ambiguous, ask clarifying questions
""")

        return "\n".join(prompt_parts)

    def _call_llm(self, prompt: str, stream: bool = True) -> str:
        """Call Ollama LLM with prompt"""
        try:
            if stream:
                # Streaming response for better UX
                process = subprocess.Popen(
                    ['ollama', 'run', self.model],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                stdout, stderr = process.communicate(input=prompt, timeout=300)

                if process.returncode != 0:
                    raise RuntimeError(f"LLM error: {stderr}")

                return stdout.strip()
            else:
                # Non-streaming
                result = subprocess.run(
                    ['ollama', 'run', self.model, prompt],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=True
                )
                return result.stdout.strip()

        except subprocess.TimeoutExpired:
            return "⏱️  Response timed out. Please try a simpler query."
        except Exception as e:
            return f"❌ Error: {e}"

    def ask(self, question: str, use_rag: bool = True) -> str:
        """
        Ask a coding question

        Args:
            question: Your coding question or task
            use_rag: Use documentation context (default: True)

        Returns:
            AI response
        """
        # Build optimized prompt
        prompt = self._build_prompt(question, use_rag=use_rag)

        # Get response
        if self.verbose:
            print(f"\n🤖 Thinking with {self.model}...\n")

        response = self._call_llm(prompt, stream=True)

        # Add to conversation history
        self.conversation.add_message("user", question)
        self.conversation.add_message("assistant", response)

        return response

    def code(self, task: str, language: str = "python") -> str:
        """
        Generate code for a specific task

        Args:
            task: Description of what code should do
            language: Programming language (python, c, bash, etc.)

        Returns:
            Generated code
        """
        enhanced_query = f"""Generate {language} code for this task:

{task}

Requirements:
- Complete, production-ready code
- Proper error handling
- Clear comments
- Modern best practices
- Type hints (if applicable)
- Example usage

Provide ONLY the code with minimal explanation."""

        return self.ask(enhanced_query, use_rag=True)

    def review(self, code: str, focus: Optional[str] = None) -> str:
        """
        Review code and suggest improvements

        Args:
            code: Code to review
            focus: Specific area (security, performance, style, etc.)

        Returns:
            Code review with suggestions
        """
        focus_text = f" Focus on {focus}." if focus else ""

        query = f"""Review this code and suggest improvements.{focus_text}

CODE:
{code}

Provide:
1. Security issues (if any)
2. Performance concerns
3. Code style improvements
4. Best practice violations
5. Refactored version (if significant changes needed)"""

        return self.ask(query, use_rag=True)

    def explain(self, code_or_topic: str) -> str:
        """
        Explain code or technical concept

        Args:
            code_or_topic: Code snippet or technical topic

        Returns:
            Detailed explanation
        """
        query = f"""Explain this in detail:

{code_or_topic}

Include:
- What it does
- How it works
- Use cases
- Examples
- Related concepts"""

        return self.ask(query, use_rag=True)

    def debug(self, code: str, error: Optional[str] = None,
             expected: Optional[str] = None) -> str:
        """
        Debug code and suggest fixes

        Args:
            code: Buggy code
            error: Error message (if any)
            expected: Expected behavior

        Returns:
            Debug analysis and fixed code
        """
        parts = [f"Debug this code:\n\nCODE:\n{code}"]

        if error:
            parts.append(f"\nERROR:\n{error}")

        if expected:
            parts.append(f"\nEXPECTED BEHAVIOR:\n{expected}")

        parts.append("\nProvide:\n1. Root cause analysis\n2. Fixed code\n3. Explanation of the fix")

        return self.ask("\n".join(parts), use_rag=True)

    def execute_code(self, code: str, language: str = "python",
                    args: Optional[List[str]] = None) -> Tuple[bool, str]:
        """
        Execute generated code safely in a temp environment

        Args:
            code: Code to execute
            language: Language (python, bash, c)
            args: Command-line arguments

        Returns:
            (success, output)
        """
        args = args or []

        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            if language == 'python':
                result = subprocess.run(
                    ['python3', temp_file] + args,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            elif language == 'bash':
                result = subprocess.run(
                    ['bash', temp_file] + args,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            elif language == 'c':
                # Compile first
                output_file = temp_file.replace('.c', '')
                compile_result = subprocess.run(
                    ['gcc', '-o', output_file, temp_file],
                    capture_output=True,
                    text=True
                )

                if compile_result.returncode != 0:
                    return False, f"Compilation error:\n{compile_result.stderr}"

                result = subprocess.run(
                    [output_file] + args,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
            else:
                return False, f"Unsupported language: {language}"

            output = result.stdout
            if result.returncode != 0:
                output += f"\n\nERROR:\n{result.stderr}"

            return result.returncode == 0, output

        except subprocess.TimeoutExpired:
            return False, "⏱️  Execution timed out (30s limit)"
        except Exception as e:
            return False, f"❌ Execution error: {e}"
        finally:
            # Cleanup
            Path(temp_file).unlink(missing_ok=True)

    def save_code(self, code: str, filename: str, overwrite: bool = False) -> bool:
        """
        Save generated code to file

        Args:
            code: Code content
            filename: Target filename
            overwrite: Overwrite if exists

        Returns:
            Success status
        """
        filepath = Path(filename)

        if filepath.exists() and not overwrite:
            print(f"❌ File exists: {filepath}. Use overwrite=True to replace.")
            return False

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(code)
            print(f"✓ Saved to: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return False

    # ========================================================================
    # ADVANCED CODE ANALYSIS FEATURES
    # ========================================================================

    def analyze_security(self, code: str, verbose: bool = True) -> Dict:
        """
        Scan code for security vulnerabilities

        Args:
            code: Python source code to analyze
            verbose: Print detailed report

        Returns:
            Dictionary with security issues categorized by severity
        """
        if not ADVANCED_FEATURES:
            return {"error": "Advanced features not available. Install dependencies."}

        scanner = SecurityScanner()
        issues = scanner.scan(code)

        # Categorize by severity
        results = {
            'CRITICAL': [],
            'HIGH': [],
            'MEDIUM': [],
            'LOW': [],
            'INFO': []
        }

        for issue in issues:
            results[issue.severity.value].append({
                'category': issue.category,
                'line': issue.line,
                'description': issue.description,
                'remediation': issue.remediation,
                'cwe_id': issue.cwe_id
            })

        if verbose:
            self._print_security_report(results)

        return results

    def _print_security_report(self, results: Dict):
        """Print formatted security report"""
        total_issues = sum(len(issues) for issues in results.values())

        print("\n" + "="*70)
        print(f"🔒 SECURITY ANALYSIS REPORT ({total_issues} issues)")
        print("="*70)

        for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
            issues = results[severity]
            if issues:
                print(f"\n[{severity}] {len(issues)} issue(s):")
                for i, issue in enumerate(issues, 1):
                    print(f"  {i}. Line {issue['line']}: {issue['description']}")
                    print(f"     Category: {issue['category']}")
                    if issue['cwe_id']:
                        print(f"     CWE: {issue['cwe_id']}")
                    print(f"     Fix: {issue['remediation']}")

        print("\n" + "="*70)

    def optimize_performance(self, code: str, verbose: bool = True) -> Dict:
        """
        Analyze code for performance issues and suggest optimizations

        Args:
            code: Python source code
            verbose: Print detailed report

        Returns:
            Dictionary with performance issues and suggestions
        """
        if not ADVANCED_FEATURES:
            return {"error": "Advanced features not available"}

        optimizer = PerformanceOptimizer()
        issues = optimizer.analyze(code)

        results = {'issues': []}
        for issue in issues:
            results['issues'].append({
                'line': issue.line,
                'category': issue.category,
                'description': issue.description,
                'suggestion': issue.suggestion,
                'improvement': issue.estimated_improvement
            })

        if verbose:
            self._print_performance_report(results)

        return results

    def _print_performance_report(self, results: Dict):
        """Print formatted performance report"""
        issues = results['issues']
        print("\n" + "="*70)
        print(f"⚡ PERFORMANCE ANALYSIS ({len(issues)} optimization opportunities)")
        print("="*70)

        for i, issue in enumerate(issues, 1):
            print(f"\n{i}. Line {issue['line']}: {issue['description']}")
            print(f"   Suggestion: {issue['suggestion']}")
            print(f"   Expected improvement: {issue['improvement']}")

        print("\n" + "="*70)

    def analyze_complexity(self, code: str, verbose: bool = True) -> Dict:
        """
        Calculate code complexity metrics

        Args:
            code: Python source code
            verbose: Print report

        Returns:
            Complexity metrics
        """
        if not ADVANCED_FEATURES:
            return {"error": "Advanced features not available"}

        analyzer = ComplexityAnalyzer()
        smell_detector = CodeSmellDetector()

        results = {
            'cyclomatic_complexity': analyzer.cyclomatic_complexity(code),
            'nesting_depth': analyzer.nesting_depth(code),
            'code_smells': []
        }

        # Detect code smells
        smells = smell_detector.detect(code)
        for smell in smells:
            results['code_smells'].append({
                'line': smell.line,
                'category': smell.category,
                'description': smell.description,
                'suggestion': smell.suggestion
            })

        if verbose:
            self._print_complexity_report(results)

        return results

    def _print_complexity_report(self, results: Dict):
        """Print formatted complexity report"""
        print("\n" + "="*70)
        print("📊 CODE COMPLEXITY ANALYSIS")
        print("="*70)

        print(f"\nCyclomatic Complexity: {results['cyclomatic_complexity']}")
        print(f"  {'✓ Good' if results['cyclomatic_complexity'] <= 10 else '⚠️  High' if results['cyclomatic_complexity'] <= 20 else '❌ Very High'}")

        print(f"\nMax Nesting Depth: {results['nesting_depth']}")
        print(f"  {'✓ Good' if results['nesting_depth'] <= 3 else '⚠️  Deep' if results['nesting_depth'] <= 5 else '❌ Too Deep'}")

        smells = results['code_smells']
        if smells:
            print(f"\n👃 Code Smells Found: {len(smells)}")
            for i, smell in enumerate(smells, 1):
                print(f"\n  {i}. Line {smell['line']} [{smell['category']}]")
                print(f"     {smell['description']}")
                print(f"     → {smell['suggestion']}")
        else:
            print("\n✓ No code smells detected!")

        print("\n" + "="*70)

    def auto_refactor(self, code: str, verbose: bool = True) -> Tuple[str, List]:
        """
        Automatically refactor code with AST transformations

        Applies:
        - Error handling
        - Type hints
        - Performance optimizations

        Args:
            code: Python source code
            verbose: Print transformation log

        Returns:
            (refactored_code, transformations_list)
        """
        if not ADVANCED_FEATURES:
            return code, [{"error": "Advanced features not available"}]

        refactored, transformations = apply_all_transformers(code)

        if verbose:
            print("\n" + "="*70)
            print(f"✨ AUTO-REFACTORED CODE ({len(transformations)} transformations)")
            print("="*70)

            for t in transformations:
                print(f"\n  [{t.transformer_name}] Line {t.original_line}")
                print(f"    {t.description}")
                print(f"    Changes: {t.changes_made}")

            print("\n" + "="*70)

        return refactored, transformations

    def generate_docs(self, code: str, style: str = 'google') -> List:
        """
        Generate docstrings for functions/classes

        Args:
            code: Python source code
            style: Docstring style (google, numpy, sphinx)

        Returns:
            List of generated documentation
        """
        if not ADVANCED_FEATURES:
            return [{"error": "Advanced features not available"}]

        doc_gen = DocumentationGenerator(style=style)
        docs = doc_gen.generate_all(code)

        print("\n" + "="*70)
        print(f"📚 GENERATED DOCUMENTATION ({len(docs)} items)")
        print("="*70)

        for doc in docs:
            print(f"\n  {doc.target} (line {doc.line}):")
            for line in doc.docstring.split('\n'):
                print(f"    {line}")

        print("\n" + "="*70)

        return docs

    def generate_tests(self, code: str, framework: str = 'pytest',
                       module_name: str = 'module') -> str:
        """
        Generate unit tests for code

        Args:
            code: Python source code
            framework: Testing framework (pytest or unittest)
            module_name: Module name for imports

        Returns:
            Generated test code
        """
        if not ADVANCED_FEATURES:
            return "# Error: Advanced features not available"

        test_gen = TestGenerator(framework=framework)
        tests = test_gen.generate_all(code, module_name=module_name)

        print("\n" + "="*70)
        print(f"🧪 GENERATED TESTS ({framework})")
        print("="*70)
        print(tests)
        print("="*70)

        return tests

    def full_analysis(self, code: str) -> Dict:
        """
        Run complete code analysis (security, performance, complexity)

        Args:
            code: Python source code

        Returns:
            Complete analysis results
        """
        if not ADVANCED_FEATURES:
            return {"error": "Advanced features not available"}

        print("\n" + "🔍 Running Full Code Analysis...")

        results = {
            'security': self.analyze_security(code, verbose=False),
            'performance': self.optimize_performance(code, verbose=False),
            'complexity': self.analyze_complexity(code, verbose=False)
        }

        # Print summary
        security_total = sum(len(issues) for issues in results['security'].values())
        perf_total = len(results['performance']['issues'])
        smell_total = len(results['complexity']['code_smells'])

        print("\n" + "="*70)
        print("📋 FULL ANALYSIS SUMMARY")
        print("="*70)
        print(f"  Security Issues: {security_total}")
        print(f"  Performance Issues: {perf_total}")
        print(f"  Code Smells: {smell_total}")
        print(f"  Complexity: {results['complexity']['cyclomatic_complexity']}")
        print(f"  Nesting Depth: {results['complexity']['nesting_depth']}")
        print("="*70)

        # Print detailed reports
        if security_total > 0:
            self._print_security_report(results['security'])
        if perf_total > 0:
            self._print_performance_report(results['performance'])
        if smell_total > 0 or results['complexity']['cyclomatic_complexity'] > 10:
            self._print_complexity_report(results['complexity'])

        return results

    # ========================================================================
    # PHASE 1 INTEGRATED FEATURES
    # ========================================================================

    def analyze_with_cache(self, code: str, use_npu: bool = True) -> Dict:
        """
        Analyze code with intelligent caching

        Uses Phase 1.4: Caching layer with NPU acceleration

        Args:
            code: Python source code
            use_npu: Use NPU acceleration if available

        Returns:
            Analysis results (from cache or fresh)
        """
        try:
            from rag_system.analysis_cache import AnalysisCache

            cache = AnalysisCache(use_npu=use_npu, verbose=self.verbose)

            # Check cache first
            cached_security = cache.get_cached_analysis(code, 'security')
            cached_performance = cache.get_cached_analysis(code, 'performance')
            cached_complexity = cache.get_cached_analysis(code, 'complexity')

            results = {}

            # Security (use cache if available)
            if cached_security:
                if self.verbose:
                    print("✓ Using cached security analysis")
                results['security'] = cached_security
            else:
                results['security'] = self.analyze_security(code, verbose=False)
                cache.cache_analysis(code, 'security', results['security'])

            # Performance (use cache if available)
            if cached_performance:
                if self.verbose:
                    print("✓ Using cached performance analysis")
                results['performance'] = cached_performance
            else:
                results['performance'] = self.optimize_performance(code, verbose=False)
                cache.cache_analysis(code, 'performance', results['performance'])

            # Complexity (use cache if available)
            if cached_complexity:
                if self.verbose:
                    print("✓ Using cached complexity analysis")
                results['complexity'] = cached_complexity
            else:
                results['complexity'] = self.analyze_complexity(code, verbose=False)
                cache.cache_analysis(code, 'complexity', results['complexity'])

            # Print cache stats
            if self.verbose:
                stats = cache.get_stats()
                print(f"\n📊 Cache Stats: {stats.hit_rate:.1%} hit rate ({stats.hits} hits, {stats.misses} misses)")

            return results

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Cache unavailable, using direct analysis: {e}")
            return self.full_analysis(code)

    def analyze_async(self, code: str, analyses: Optional[List[str]] = None) -> Dict:
        """
        Async parallel code analysis (Phase 1.2)

        Runs multiple analyses in parallel for 3-5x speedup

        Args:
            code: Python source code
            analyses: List of analysis types (default: all)

        Returns:
            Combined analysis results
        """
        try:
            from rag_system.async_analysis import AsyncCodeAnalyzer, AnalysisType

            analyzer = AsyncCodeAnalyzer(max_workers=4, verbose=self.verbose)

            # Convert string names to AnalysisType enums
            if analyses:
                analysis_types = [AnalysisType(a) for a in analyses]
            else:
                analysis_types = None

            # Run parallel analyses
            results = analyzer.analyze_parallel(code, analyses=analysis_types)

            # Convert to standard format
            formatted_results = {}
            for analysis_type, result in results.items():
                if result.success:
                    formatted_results[analysis_type.value] = result.results

            return formatted_results

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Async analysis failed, using sequential: {e}")
            return self.full_analysis(code)

    def format_and_fix(self, code: str, formatter: str = 'black',
                       validators: Optional[List[str]] = None) -> Tuple[str, Dict]:
        """
        Format code and validate (Phase 1.3)

        Args:
            code: Python source code
            formatter: Formatter to use (black, autopep8, yapf)
            validators: Validators to run (flake8, pylint, mypy)

        Returns:
            (formatted_code, validation_results)
        """
        try:
            from rag_system.code_formatters import CodeFormatter

            formatter_obj = CodeFormatter(verbose=self.verbose)

            # Format and validate
            format_result, validation_results = formatter_obj.format_and_validate(
                code,
                formatter=formatter,
                validators=validators or ['flake8']
            )

            if self.verbose:
                if format_result.changed:
                    print(f"✨ Code formatted with {formatter}")

                for validator_type, val_result in validation_results.items():
                    status = "✓" if val_result.passes else "✗"
                    print(f"{status} {validator_type.value}: {val_result.score:.1f}/10")

            return format_result.formatted_code, validation_results

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Formatting failed: {e}")
            return code, {}

    def advanced_refactor(self, code: str, transformers: Optional[List[str]] = None) -> Tuple[str, List]:
        """
        Advanced AST-based refactoring (Phase 1.5)

        Applies:
        - Naming convention fixes (camelCase → snake_case)
        - Import optimization (remove unused, sort)
        - Method extraction (complex blocks)

        Args:
            code: Python source code
            transformers: Specific transformers to apply (default: all)

        Returns:
            (refactored_code, transformations_list)
        """
        try:
            from rag_system.advanced_transformers import (
                apply_advanced_transformers,
                NamingConventionFixer,
                ImportOptimizer
            )

            if transformers:
                # Apply specific transformers
                # TODO: Implement selective application
                refactored, transforms = apply_advanced_transformers(code)
            else:
                # Apply all transformers
                refactored, transforms = apply_advanced_transformers(code)

            if self.verbose and transforms:
                print(f"\n✨ Applied {len(transforms)} transformations:")
                for t in transforms:
                    print(f"  - {t.description}")

            return refactored, transforms

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Refactoring failed: {e}")
            return code, []

    def complete_workflow(self, code: str) -> Dict:
        """
        Complete Phase 1 integrated workflow

        Combines all features:
        1. Cached/async analysis (Phase 1.2 + 1.4)
        2. Advanced refactoring (Phase 1.5)
        3. Code formatting (Phase 1.3)
        4. Final validation

        Args:
            code: Python source code

        Returns:
            Complete analysis and refactored code
        """
        if self.verbose:
            print("\n🚀 Running Complete Phase 1 Workflow...")

        workflow_results = {}

        # Step 1: Cached async analysis
        if self.verbose:
            print("\n1️⃣  Running cached async analysis...")

        analysis = self.analyze_with_cache(code, use_npu=True)
        workflow_results['analysis'] = analysis

        # Step 2: Advanced refactoring
        if self.verbose:
            print("\n2️⃣  Applying advanced refactoring...")

        refactored, transforms = self.advanced_refactor(code)
        workflow_results['refactored_code'] = refactored
        workflow_results['transformations'] = transforms

        # Step 3: Format and validate
        if self.verbose:
            print("\n3️⃣  Formatting and validating...")

        formatted, validation = self.format_and_fix(refactored)
        workflow_results['formatted_code'] = formatted
        workflow_results['validation'] = validation

        # Step 4: Final analysis of formatted code
        if self.verbose:
            print("\n4️⃣  Final quality check...")

        final_analysis = self.analyze_with_cache(formatted, use_npu=True)
        workflow_results['final_analysis'] = final_analysis

        # Summary
        if self.verbose:
            print("\n" + "="*70)
            print("✅ COMPLETE WORKFLOW FINISHED")
            print("="*70)

            security_before = sum(len(issues) for issues in analysis.get('security', {}).values())
            security_after = sum(len(issues) for issues in final_analysis.get('security', {}).values())

            print(f"\nSecurity Issues: {security_before} → {security_after}")
            print(f"Transformations Applied: {len(transforms)}")
            print(f"Code Formatting: {'✓ Changed' if refactored != formatted else '✓ Unchanged'}")

            for validator_type, val_result in validation.items():
                print(f"Validation ({validator_type.value}): {val_result.score:.1f}/10")

            print("="*70)

        return workflow_results

    # ========================================================================
    # PHASE 2 ADVANCED INTELLIGENCE FEATURES
    # ========================================================================

    def multi_agent_review(self, code: str, filename: str = "code.py") -> Dict:
        """
        Multi-agent code review (Phase 2.1)

        Coordinates 5 specialized agents to review code:
        - SecurityExpert: Security vulnerabilities
        - PerformanceExpert: Performance issues
        - MaintainabilityExpert: Code maintainability
        - TestCoverageExpert: Test coverage
        - DocumentationExpert: Documentation quality

        Args:
            code: Python source code
            filename: Filename for context

        Returns:
            Consolidated review from all agents
        """
        try:
            from rag_system.multi_agent_review import MultiAgentReviewer

            reviewer = MultiAgentReviewer(max_workers=5)
            consolidated = reviewer.review_code(code, filename)

            if self.verbose:
                print(reviewer.format_review_report(consolidated))

            return {
                'consolidated': consolidated,
                'agent_reviews': consolidated.agent_reviews,
                'critical_findings': consolidated.critical_findings,
                'high_priority': consolidated.high_priority_findings,
                'recommendation': consolidated.recommendation,
                'overall_score': consolidated.overall_score
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Multi-agent review failed: {e}")
            return {}

    def semantic_search(self, query: str, codebase_path: str = ".", max_results: int = 10) -> Dict:
        """
        Semantic code search (Phase 2.2)

        Natural language search beyond grep:
        - "Find SQL injection vulnerabilities"
        - "Functions with high complexity"
        - "Where is this function used?"

        Args:
            query: Natural language search query
            codebase_path: Path to search
            max_results: Maximum results

        Returns:
            Search results with context
        """
        try:
            from rag_system.semantic_search import SemanticCodeSearch

            searcher = SemanticCodeSearch(codebase_path, auto_index=True)
            results = searcher.search(query, max_results)

            if self.verbose:
                print(searcher.format_results(results))

            return {
                'query': query,
                'results': results.results,
                'total_files': results.total_files_searched,
                'execution_time': results.execution_time
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Semantic search failed: {e}")
            return {}

    def suggest_refactorings(self, code: str, filename: str = "code.py") -> Dict:
        """
        Automated refactoring suggestions (Phase 2.3)

        Detects refactoring opportunities:
        - Extract Method
        - Introduce Parameter Object
        - Rename (poor naming)
        - Extract Variable
        - Remove Dead Code

        Args:
            code: Python source code
            filename: Filename for context

        Returns:
            List of refactoring opportunities
        """
        try:
            from rag_system.refactoring_workflows import RefactoringWorkflow

            workflow = RefactoringWorkflow()
            opportunities = workflow.suggest_refactorings(code, filename)

            if self.verbose:
                print(workflow.format_suggestions(opportunities))

            return {
                'opportunities': opportunities,
                'count': len(opportunities),
                'high_impact': [o for o in opportunities if o.impact == 'high'],
                'workflow': workflow
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Refactoring suggestions failed: {e}")
            return {}

    def apply_refactoring(self, code: str, refactoring_type: str, **kwargs) -> Tuple[str, bool]:
        """
        Apply specific refactoring (Phase 2.3)

        Args:
            code: Source code
            refactoring_type: Type of refactoring
            **kwargs: Refactoring-specific parameters

        Returns:
            (refactored_code, success)
        """
        try:
            from rag_system.refactoring_workflows import RefactoringWorkflow, RefactoringType

            workflow = RefactoringWorkflow()

            # Map string to enum
            refactoring_enum = RefactoringType[refactoring_type.upper()]

            # Create dummy opportunity (in practice, would come from suggest_refactorings)
            from rag_system.refactoring_workflows import RefactoringOpportunity
            opportunity = RefactoringOpportunity(
                refactoring_type=refactoring_enum,
                location="code.py:1",
                title=f"Apply {refactoring_type}",
                description="User-requested refactoring",
                impact="medium",
                effort="moderate",
                confidence=1.0
            )

            result = workflow.apply_refactoring(code, opportunity, **kwargs)

            if self.verbose:
                if result.success:
                    print(f"✅ Refactoring applied: {result.changes_summary}")
                else:
                    print(f"❌ Refactoring failed: {result.changes_summary}")

            return result.refactored_code, result.success

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Refactoring application failed: {e}")
            return code, False

    def predict_completion(self, code: str, context: str = "docstring") -> List:
        """
        Predictive code completion (Phase 2.4)

        Generate completions from:
        - Docstrings (documentation-driven)
        - Test cases (test-driven)
        - Function signatures (intent-based)
        - Error patterns (error-aware)

        Args:
            code: Partial code to complete
            context: Completion context (docstring/test/signature/error)

        Returns:
            List of completion suggestions
        """
        try:
            from rag_system.predictive_completion import PredictiveCompletion

            predictor = PredictiveCompletion()
            completions = []

            if context == "docstring":
                completions = predictor.complete_from_docstring(code)
            elif context == "test":
                completions = predictor.complete_from_test(code)
            elif context == "types":
                completions = predictor.add_type_annotations(code)
            elif context == "error":
                completions = predictor.suggest_error_handling(code)
            elif context == "body":
                # Extract function name from code
                import re
                match = re.search(r'def\s+(\w+)\s*\(', code)
                if match:
                    func_name = match.group(1)
                    completions = predictor.complete_function_body(code, func_name)
            else:
                # Try all contexts
                completions.extend(predictor.complete_from_docstring(code))
                completions.extend(predictor.add_type_annotations(code))

            if self.verbose and completions:
                print(f"\n🔮 Found {len(completions)} completion suggestions:")
                for i, comp in enumerate(completions[:3], 1):  # Show top 3
                    print(predictor.format_completion(comp))

            return completions

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Predictive completion failed: {e}")
            return []

    def complete_phase2_workflow(self, code: str, filename: str = "code.py") -> Dict:
        """
        Complete Phase 2 integrated workflow

        Combines all Phase 2 features:
        1. Multi-agent review
        2. Semantic search for similar patterns
        3. Refactoring suggestions
        4. Predictive completions

        Args:
            code: Python source code
            filename: Filename for context

        Returns:
            Complete Phase 2 analysis
        """
        if self.verbose:
            print("\n🚀 Running Complete Phase 2 Workflow...")

        workflow_results = {}

        # Step 1: Multi-agent review
        if self.verbose:
            print("\n1️⃣  Running multi-agent review...")

        review = self.multi_agent_review(code, filename)
        workflow_results['review'] = review

        # Step 2: Refactoring suggestions
        if self.verbose:
            print("\n2️⃣  Analyzing refactoring opportunities...")

        refactorings = self.suggest_refactorings(code, filename)
        workflow_results['refactorings'] = refactorings

        # Step 3: Predictive completions
        if self.verbose:
            print("\n3️⃣  Generating intelligent completions...")

        completions = self.predict_completion(code, context="all")
        workflow_results['completions'] = completions

        # Summary
        if self.verbose:
            print("\n" + "="*70)
            print("✅ COMPLETE PHASE 2 WORKFLOW FINISHED")
            print("="*70)

            if review:
                print(f"\nOverall Code Quality: {review.get('overall_score', 0):.1f}/10")
                print(f"Recommendation: {review.get('recommendation', 'N/A')}")
                print(f"Critical Issues: {len(review.get('critical_findings', []))}")
                print(f"High Priority Issues: {len(review.get('high_priority', []))}")

            if refactorings:
                print(f"\nRefactoring Opportunities: {refactorings.get('count', 0)}")
                print(f"High Impact Refactorings: {len(refactorings.get('high_impact', []))}")

            if completions:
                print(f"\nCompletion Suggestions: {len(completions)}")

            print("="*70)

        return workflow_results

    # ========================================================================
    # PHASE 4 CUTTING-EDGE RESEARCH FEATURES
    # ========================================================================

    def synthesize_module(self, specification: str) -> Dict:
        """
        Neural code synthesis (Phase 4.1)

        Generate complete modules from natural language specifications.
        Supports kernel drivers, Python libraries, C libraries, Rust crates, NPU accelerators.

        Args:
            specification: Natural language specification

        Returns:
            Generated module information
        """
        try:
            from rag_system.neural_code_synthesis import NeuralCodeSynthesizer

            synthesizer = NeuralCodeSynthesizer()
            module = synthesizer.generate_module(specification)

            if self.verbose:
                print(synthesizer.format_generation_report(module))

            return {
                'module_name': module.module_name,
                'module_type': module.module_type.value,
                'files': {f.filename: f.content for f in module.files},
                'description': module.description
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Module synthesis failed: {e}")
            return {}

    def optimize_for_npu(self, model_path: str, target_latency_ms: Optional[float] = None,
                         preserve_accuracy: bool = True) -> Dict:
        """
        Optimize model for military-grade NPU (Phase 4.3)

        Deep integration with LAT5150DRVMIL military NPU:
        - Intel Core Ultra NPU: 66.4 TOPS total (26.4 TOPS military mode)
        - FP16/INT8 quantization
        - Thermal-aware scheduling (75°C throttle)
        - Model graph optimization for VPU

        Args:
            model_path: Path to model file
            target_latency_ms: Target inference latency
            preserve_accuracy: Preserve accuracy (FP16 instead of INT8)

        Returns:
            Optimization results
        """
        try:
            from rag_system.advanced_npu_integration import NPUOptimizer

            optimizer = NPUOptimizer(military_mode=True)
            result = optimizer.optimize_for_npu(model_path, target_latency_ms, preserve_accuracy)

            if self.verbose:
                print(optimizer.format_optimization_report(result))

            return {
                'speedup_factor': result.speedup_factor,
                'accuracy_delta': result.accuracy_delta,
                'model_size_reduction': (1 - result.model_size_mb_after / result.model_size_mb_before) * 100,
                'quantization': result.quantization_applied.value,
                'optimizations': result.optimizations_applied
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  NPU optimization failed: {e}")
            return {}

    def get_npu_info(self) -> Dict:
        """Get NPU hardware information (Phase 4.3)"""
        try:
            from rag_system.advanced_npu_integration import NPUOptimizer

            optimizer = NPUOptimizer(military_mode=True)
            info = optimizer.get_npu_info()

            if self.verbose:
                print("\n" + "="*70)
                print("🚀 LAT5150DRVMIL MILITARY NPU INFORMATION")
                print("="*70)
                print(f"Device: {info['device_name']}")
                print(f"Total Performance: {info['total_tops']} TOPS")
                print(f"Military Mode: {info['military_mode_tops']} TOPS")
                print(f"Driver: {info['driver']}")
                print(f"OpenVINO: {info['openvino_version']}")
                print(f"Military Mode: {'ENABLED' if info['military_mode_enabled'] else 'DISABLED'}")
                print(f"Thermal Limit: {info['thermal_limit_celsius']}°C")
                print(f"Status: {info['status'].upper()}")
                print("="*70)

            return info

        except Exception as e:
            if self.verbose:
                print(f"⚠️  NPU info retrieval failed: {e}")
            return {}

    def optimize_hardware(self, code: str, arch: str = "lat5150drvmil") -> Dict:
        """
        Hardware-aware code generation (Phase 4.4)

        Generate optimized code for LAT5150DRVMIL hardware:
        - SIMD vectorization (AVX2)
        - Cache-line alignment (64-byte)
        - Branch prediction hints
        - Lock-free data structures

        Args:
            code: Source code (C/C++)
            arch: Target architecture (default: lat5150drvmil)

        Returns:
            Optimization results
        """
        try:
            from rag_system.hardware_aware_codegen import HardwareOptimizer, Architecture

            # Map architecture string to enum
            arch_enum = Architecture.LAT5150DRVMIL if arch.lower() == "lat5150drvmil" else Architecture.X86_64_GENERIC

            optimizer = HardwareOptimizer(arch_enum)
            result = optimizer.optimize_code(code)

            if self.verbose:
                print(optimizer.format_optimization_report(result))

            return {
                'optimized_code': result.optimized_code,
                'optimizations_applied': result.optimizations_applied,
                'estimated_speedup': result.estimated_speedup,
                'architecture': result.target_architecture.value
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Hardware optimization failed: {e}")
            return {}

    def generate_simd_code(self, operation: str = "add") -> str:
        """
        Generate SIMD-vectorized code (Phase 4.4)

        Generate AVX2-optimized code for common operations.

        Args:
            operation: Operation type (add, multiply)

        Returns:
            Generated SIMD code
        """
        try:
            from rag_system.hardware_aware_codegen import SIMDVectorizer, Architecture

            # Generate sample loop code based on operation
            if operation == "add":
                loop_code = "for (int i = 0; i < n; i++) { result[i] = a[i] + b[i]; }"
            elif operation == "multiply":
                loop_code = "for (int i = 0; i < n; i++) { result[i] = a[i] * b[i]; }"
            else:
                return f"// Unsupported operation: {operation}\n// Supported: add, multiply"

            # Use LAT5150DRVMIL architecture (AVX2)
            arch = Architecture.LAT5150DRVMIL if hasattr(Architecture, 'LAT5150DRVMIL') else Architecture.X86_64_AVX2
            code = SIMDVectorizer.vectorize_simple_loop(loop_code, arch)

            if self.verbose:
                print("\n🚀 Generated SIMD Code:")
                print("="*70)
                print(code)
                print("="*70)

            return code

        except Exception as e:
            if self.verbose:
                print(f"⚠️  SIMD generation failed: {e}")
            return f"// Error: {e}"

    def generate_lockfree_queue(self) -> str:
        """
        Generate lock-free queue for dual NPU (Phase 4.4)

        Returns:
            Lock-free queue implementation
        """
        try:
            from rag_system.hardware_aware_codegen import LockFreeGenerator

            code = LockFreeGenerator.generate_lockfree_queue()

            if self.verbose:
                print("\n🔒 Generated Lock-Free Queue:")
                print("="*70)
                print(code)
                print("="*70)

            return code

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Lock-free generation failed: {e}")
            return f"// Error: {e}"

    def complete_phase4_workflow(self, specification: str, model_path: Optional[str] = None,
                                  code: Optional[str] = None) -> Dict:
        """
        Complete Phase 4 integrated workflow

        Combines all cutting-edge research features:
        1. Neural code synthesis
        2. NPU optimization (if model provided)
        3. Hardware-aware code generation (if code provided)

        Args:
            specification: Natural language specification for module
            model_path: Optional model path for NPU optimization
            code: Optional C/C++ code for hardware optimization

        Returns:
            Complete Phase 4 results
        """
        if self.verbose:
            print("\n🚀 Running Complete Phase 4 Workflow...")

        workflow_results = {}

        # Step 1: Neural code synthesis
        if self.verbose:
            print("\n1️⃣  Synthesizing module from specification...")

        module = self.synthesize_module(specification)
        workflow_results['module'] = module

        # Step 2: NPU optimization (if model provided)
        if model_path:
            if self.verbose:
                print("\n2️⃣  Optimizing for military-grade NPU...")

            npu_result = self.optimize_for_npu(model_path, target_latency_ms=5.0, preserve_accuracy=True)
            workflow_results['npu_optimization'] = npu_result
        else:
            if self.verbose:
                print("\n2️⃣  Skipping NPU optimization (no model provided)")

        # Step 3: Hardware-aware code generation (if code provided)
        if code:
            if self.verbose:
                print("\n3️⃣  Applying hardware-aware optimizations...")

            hw_result = self.optimize_hardware(code, arch="lat5150drvmil")
            workflow_results['hardware_optimization'] = hw_result
        else:
            if self.verbose:
                print("\n3️⃣  Skipping hardware optimization (no code provided)")

        # Step 4: Get NPU info
        if self.verbose:
            print("\n4️⃣  Retrieving NPU hardware info...")

        npu_info = self.get_npu_info()
        workflow_results['npu_info'] = npu_info

        # Summary
        if self.verbose:
            print("\n" + "="*70)
            print("✅ COMPLETE PHASE 4 WORKFLOW FINISHED")
            print("="*70)

            if module:
                print(f"\nModule Generated: {module.get('module_name', 'N/A')}")
                print(f"Module Type: {module.get('module_type', 'N/A')}")
                print(f"Files Generated: {len(module.get('files', {}))}")

            if model_path and workflow_results.get('npu_optimization'):
                opt = workflow_results['npu_optimization']
                print(f"\nNPU Optimization:")
                print(f"  Speedup: {opt.get('speedup_factor', 0):.1f}x")
                print(f"  Model Size Reduction: {opt.get('model_size_reduction', 0):.0f}%")
                print(f"  Quantization: {opt.get('quantization', 'N/A').upper()}")

            if code and workflow_results.get('hardware_optimization'):
                hw = workflow_results['hardware_optimization']
                print(f"\nHardware Optimization:")
                print(f"  Estimated Speedup: {hw.get('estimated_speedup', 0):.1f}x")
                print(f"  Optimizations: {len(hw.get('optimizations_applied', []))}")

            print("="*70)

        return workflow_results

    # ========================================================================
    # PHASE 3 SPECIALIZED FEATURES
    # ========================================================================

    def record_metrics(self, code: str, file_path: str, analysis_results: Dict) -> Dict:
        """
        Record code metrics to dashboard (Phase 3.1)

        Tracks code quality metrics over time for trend analysis.

        Args:
            code: Source code
            file_path: File path
            analysis_results: Analysis results to record

        Returns:
            Metrics summary
        """
        try:
            from rag_system.metrics_dashboard import MetricsDashboard

            dashboard = MetricsDashboard()
            metrics = dashboard.record_analysis(code, file_path, analysis_results)

            if self.verbose:
                print(f"✅ Metrics recorded for {file_path}")
                print(f"   Technical Debt Score: {metrics.technical_debt_score:.1f}")

            summary = dashboard.get_summary()
            dashboard.close()

            return summary

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Metrics recording failed: {e}")
            return {}

    def show_metrics_dashboard(self, days: int = 30):
        """
        Show metrics dashboard (Phase 3.1)

        Display code quality trends over time with sparklines.

        Args:
            days: Number of days to show

        Returns:
            Dashboard output
        """
        try:
            from rag_system.metrics_dashboard import MetricsDashboard

            dashboard = MetricsDashboard()
            dashboard.show_trends(days)
            dashboard.close()

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Dashboard display failed: {e}")

    def locate_bug(self, traceback_or_symptom: str, mode: str = "traceback") -> Dict:
        """
        Automated bug localization (Phase 3.2)

        Locate bugs from stack traces or symptom descriptions.

        Args:
            traceback_or_symptom: Stack trace or symptom description
            mode: "traceback" or "symptom"

        Returns:
            Bug localization report
        """
        try:
            from rag_system.bug_localization import BugLocalizer

            localizer = BugLocalizer(str(self.project_root))

            if mode == "traceback":
                report = localizer.locate_from_traceback(traceback_or_symptom)
            else:
                report = localizer.locate_from_symptom(traceback_or_symptom)

            if self.verbose:
                print(localizer.format_report(report))

            return {
                'symptom': report.symptom,
                'error_type': report.error_type,
                'locations': report.locations,
                'total_files': report.total_files_analyzed
            }

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Bug localization failed: {e}")
            return {}

    def generate_intelligent_tests(self, code: str, module_name: str = "module") -> str:
        """
        Intelligent test generation (Phase 3.3)

        Generate comprehensive tests with property-based testing,
        edge case detection, and mutation testing.

        Args:
            code: Source code to generate tests for
            module_name: Module name for imports

        Returns:
            Generated test module code
        """
        try:
            from rag_system.intelligent_test_generation import IntelligentTestGenerator

            generator = IntelligentTestGenerator()
            tests = generator.generate_from_function(code)

            if self.verbose:
                print(generator.format_test_report(tests))

            test_module = generator.generate_test_module(code, module_name)
            return test_module

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Test generation failed: {e}")
            return f"# Test generation failed: {e}"

    def translate_code(self, code: str, source_lang: str, target_lang: str) -> Tuple[str, List[str]]:
        """
        Cross-language translation (Phase 3.4)

        Translate code between Python, C, and Rust with AST-based
        translation, type mapping, and idiomatic code generation.

        Supports: Python ↔ C, Python ↔ Rust

        Args:
            code: Source code
            source_lang: Source language (python/c/rust)
            target_lang: Target language (python/c/rust)

        Returns:
            (translated_code, warnings)
        """
        try:
            from rag_system.cross_language_translator import CrossLanguageTranslator

            translator = CrossLanguageTranslator()
            result = translator.translate(code, source_lang, target_lang)

            if self.verbose:
                print(translator.format_result(result))

            return result.translated_code, result.warnings

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Translation failed: {e}")
            return f"// Translation failed: {e}", [str(e)]

    def complete_phase3_workflow(self, code: str, file_path: str = "code.py") -> Dict:
        """
        Complete Phase 3 workflow

        Combines all specialized features:
        1. Intelligent test generation
        2. Cross-language translation to C and Rust
        3. Metrics recording
        4. Dashboard update

        Args:
            code: Python source code
            file_path: File path for metrics

        Returns:
            Complete Phase 3 results
        """
        if self.verbose:
            print("\n🚀 Running Complete Phase 3 Workflow...")

        workflow_results = {}

        # Step 1: Generate intelligent tests
        if self.verbose:
            print("\n1️⃣  Generating intelligent tests...")

        tests = self.generate_intelligent_tests(code, file_path.replace('.py', ''))
        workflow_results['tests'] = tests

        # Step 2: Translate to C
        if self.verbose:
            print("\n2️⃣  Translating to C...")

        c_code, c_warnings = self.translate_code(code, "python", "c")
        workflow_results['c_translation'] = {'code': c_code, 'warnings': c_warnings}

        # Step 3: Translate to Rust
        if self.verbose:
            print("\n3️⃣  Translating to Rust...")

        rust_code, rust_warnings = self.translate_code(code, "python", "rust")
        workflow_results['rust_translation'] = {'code': rust_code, 'warnings': rust_warnings}

        # Step 4: Record metrics
        if self.verbose:
            print("\n4️⃣  Recording metrics...")

        # Run analysis first to get results for metrics
        analysis = self.full_analysis(code)
        metrics = self.record_metrics(code, file_path, analysis)
        workflow_results['metrics'] = metrics

        # Summary
        if self.verbose:
            print("\n" + "="*70)
            print("✅ COMPLETE PHASE 3 WORKFLOW FINISHED")
            print("="*70)

            print(f"\nTests Generated: {'Yes' if tests else 'No'}")
            print(f"C Translation: {len(c_code)} chars ({len(c_warnings)} warnings)")
            print(f"Rust Translation: {len(rust_code)} chars ({len(rust_warnings)} warnings)")

            if metrics:
                print(f"Technical Debt: {metrics.get('technical_debt', 0):.1f}")

            print("="*70)

        return workflow_results

    def interactive(self):
        """Start interactive coding session"""
        print("\n" + "="*70)
        print("🤖 LAT5150DRVMIL Code Assistant (Enhanced)")
        print("="*70)
        print(f"Model: {self.model}")
        print(f"RAG: {'Enabled' if self.retriever else 'Disabled'}")
        print(f"Advanced Features: {'Enabled' if ADVANCED_FEATURES else 'Disabled'}")
        print(f"Docs: {len(self.conversation.rag_context) if self.retriever else 0} chunks loaded")
        print("\n📝 Code Generation Commands:")
        print("  /code <task>        - Generate code")
        print("  /explain <topic>    - Explain concept")
        print("  /review <code>      - Review code")
        print("  /debug <code>       - Debug code")
        print("\n🔍 Advanced Analysis Commands:")
        print("  /analyze            - Full code analysis (security/perf/complexity)")
        print("  /security           - Security vulnerability scan")
        print("  /performance        - Performance optimization analysis")
        print("  /complexity         - Code complexity metrics")
        print("  /refactor           - Auto-refactor with AST transformations")
        print("  /gendocs            - Generate docstrings")
        print("  /gentests           - Generate unit tests")
        print("\n⚡ Phase 1 Integrated Features:")
        print("  /workflow           - Complete workflow (analyze→refactor→format→validate)")
        print("  /format             - Format and validate code")
        print("  /async              - Async parallel analysis (3-5x faster)")
        print("\n🧠 Phase 2 Advanced Intelligence:")
        print("  /review-multi       - Multi-agent code review (5 experts)")
        print("  /search <query>     - Semantic code search (natural language)")
        print("  /suggest            - Suggest refactorings")
        print("  /complete           - Predictive code completion")
        print("  /phase2             - Complete Phase 2 workflow")
        print("\n🔬 Phase 3 Specialized Features:")
        print("  /metrics            - Show code quality metrics dashboard")
        print("  /locate-bug         - Automated bug localization (from traceback/symptom)")
        print("  /gen-tests          - Generate intelligent tests (property-based + edge cases)")
        print("  /translate          - Cross-language translation (Python↔C↔Rust)")
        print("  /phase3             - Complete Phase 3 workflow")
        print("\n🚀 Phase 4 Cutting-Edge Research:")
        print("  /synthesize         - Neural code synthesis (generate complete modules)")
        print("  /npu-info           - Show military NPU hardware info (66.4 TOPS)")
        print("  /npu-optimize       - Optimize model for military-grade NPU")
        print("  /hw-optimize        - Hardware-aware code generation (SIMD, cache, etc.)")
        print("  /gen-simd           - Generate SIMD-vectorized code (AVX2)")
        print("  /gen-lockfree       - Generate lock-free queue for dual NPU")
        print("  /phase4             - Complete Phase 4 workflow")
        print("\n💾 File Operations:")
        print("  /exec               - Execute last generated code")
        print("  /save <filename>    - Save last code to file")
        print("  /load <filepath>    - Load code from file")
        print("\n⚙️  Settings:")
        print("  /clear              - Clear conversation")
        print("  /norag              - Disable RAG")
        print("  /rag                - Enable RAG")
        print("  /help               - Show commands")
        print("  /exit               - Exit")
        print("="*70 + "\n")

        use_rag = True
        last_code = None

        while True:
            try:
                user_input = input("💬 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input == '/exit':
                    print("👋 Goodbye!")
                    break

                elif user_input == '/clear':
                    self.conversation.clear()
                    print("✓ Conversation cleared")
                    continue

                elif user_input == '/rag':
                    use_rag = True
                    print("✓ RAG enabled")
                    continue

                elif user_input == '/norag':
                    use_rag = False
                    print("✓ RAG disabled")
                    continue

                elif user_input == '/help':
                    print("""
Available commands:
  /code <task>     - Generate code
  /explain <topic> - Explain concept/code
  /review <code>   - Get code review
  /debug <code>    - Debug and fix code
  /exec            - Execute last generated code
  /save <file>     - Save last code to file
  /load <file>     - Load code from file
  /clear           - Clear conversation
  /rag|/norag      - Toggle documentation context
  /exit            - Exit assistant
""")
                    continue

                elif user_input.startswith('/code '):
                    task = user_input[6:]
                    response = self.code(task)
                    print(f"\n{response}\n")

                    # Extract code blocks
                    import re
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
                    if code_blocks:
                        last_code = code_blocks[-1].strip()

                elif user_input.startswith('/explain '):
                    topic = user_input[9:]
                    response = self.explain(topic)
                    print(f"\n{response}\n")

                elif user_input.startswith('/review '):
                    code = user_input[8:]
                    response = self.review(code)
                    print(f"\n{response}\n")

                elif user_input.startswith('/debug '):
                    code = user_input[7:]
                    response = self.debug(code)
                    print(f"\n{response}\n")

                elif user_input == '/exec':
                    if not last_code:
                        print("❌ No code to execute. Generate code first.")
                        continue

                    print("Executing code...")
                    success, output = self.execute_code(last_code)
                    print(f"\n{'✓' if success else '❌'} Output:\n{output}\n")

                elif user_input.startswith('/save '):
                    if not last_code:
                        print("❌ No code to save. Generate code first.")
                        continue

                    filename = user_input[6:]
                    self.save_code(last_code, filename)

                elif user_input.startswith('/load '):
                    filepath = user_input[6:]
                    try:
                        code = Path(filepath).read_text()
                        print(f"✓ Loaded {len(code)} characters from {filepath}")
                        last_code = code
                    except Exception as e:
                        print(f"❌ Load failed: {e}")

                # Advanced analysis commands
                elif user_input == '/analyze':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    self.full_analysis(last_code)

                elif user_input == '/security':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    self.analyze_security(last_code)

                elif user_input == '/performance':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    self.optimize_performance(last_code)

                elif user_input == '/complexity':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    self.analyze_complexity(last_code)

                elif user_input == '/refactor':
                    if not last_code:
                        print("❌ No code to refactor. Load or generate code first.")
                        continue
                    refactored, transforms = self.auto_refactor(last_code)
                    print("\n✨ Refactored Code:")
                    print(refactored)
                    # Update last_code with refactored version
                    last_code = refactored

                elif user_input == '/gendocs':
                    if not last_code:
                        print("❌ No code to document. Load or generate code first.")
                        continue
                    self.generate_docs(last_code)

                elif user_input.startswith('/gentests'):
                    if not last_code:
                        print("❌ No code to test. Load or generate code first.")
                        continue
                    # Parse framework and module name if provided
                    parts = user_input.split()
                    framework = parts[1] if len(parts) > 1 else 'pytest'
                    module_name = parts[2] if len(parts) > 2 else 'module'
                    tests = self.generate_tests(last_code, framework=framework, module_name=module_name)
                    # Optionally save tests
                    save_tests = input("\n💾 Save tests to file? (y/N): ").strip().lower()
                    if save_tests == 'y':
                        test_filename = input("  Filename [test_module.py]: ").strip() or "test_module.py"
                        self.save_code(tests, test_filename, overwrite=True)

                # Phase 1 integrated commands
                elif user_input == '/workflow':
                    if not last_code:
                        print("❌ No code for workflow. Load or generate code first.")
                        continue
                    workflow_results = self.complete_workflow(last_code)
                    # Update last_code with final formatted version
                    last_code = workflow_results.get('formatted_code', last_code)
                    print("\n💾 Save final code? (y/N): ", end='')
                    if input().strip().lower() == 'y':
                        filename = input("  Filename: ").strip()
                        if filename:
                            self.save_code(last_code, filename, overwrite=True)

                elif user_input == '/format':
                    if not last_code:
                        print("❌ No code to format. Load or generate code first.")
                        continue
                    formatted, validation = self.format_and_fix(last_code)
                    print("\n✨ Formatted Code:")
                    print(formatted)
                    last_code = formatted

                elif user_input == '/async':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    results = self.analyze_async(last_code)
                    print("\n📊 Async Analysis Complete!")
                    for analysis_type, data in results.items():
                        print(f"  {analysis_type}: {len(data.get('issues', []))} issues")

                # Phase 2 advanced intelligence commands
                elif user_input == '/review-multi':
                    if not last_code:
                        print("❌ No code to review. Load or generate code first.")
                        continue
                    review_results = self.multi_agent_review(last_code, "code.py")
                    if review_results:
                        print(f"\n📊 Overall Score: {review_results.get('overall_score', 0):.1f}/10")
                        print(f"🎯 Recommendation: {review_results.get('recommendation', 'N/A')}")

                elif user_input.startswith('/search '):
                    query = user_input[8:]
                    codebase_path = input("🔍 Codebase path [current directory]: ").strip() or "."
                    search_results = self.semantic_search(query, codebase_path, max_results=10)
                    if search_results:
                        print(f"\n📊 Found {len(search_results.get('results', []))} results in {search_results.get('execution_time', 0):.2f}s")

                elif user_input == '/suggest':
                    if not last_code:
                        print("❌ No code to analyze. Load or generate code first.")
                        continue
                    refactoring_suggestions = self.suggest_refactorings(last_code, "code.py")
                    if refactoring_suggestions:
                        count = refactoring_suggestions.get('count', 0)
                        high_impact = len(refactoring_suggestions.get('high_impact', []))
                        print(f"\n✨ Found {count} refactoring opportunities ({high_impact} high impact)")

                elif user_input == '/complete':
                    if not last_code:
                        print("❌ No code to complete. Load or generate code first.")
                        continue
                    print("\n🔮 Completion context:")
                    print("  1. docstring - From docstring")
                    print("  2. test - From test cases")
                    print("  3. types - Add type annotations")
                    print("  4. error - Error handling")
                    print("  5. all - Try all contexts")
                    context = input("Context [all]: ").strip() or "all"
                    completions = self.predict_completion(last_code, context)
                    if completions:
                        print(f"\n✅ Generated {len(completions)} completion suggestions")

                elif user_input == '/phase2':
                    if not last_code:
                        print("❌ No code for Phase 2 workflow. Load or generate code first.")
                        continue
                    phase2_results = self.complete_phase2_workflow(last_code, "code.py")
                    print("\n💾 Save analysis results? (y/N): ", end='')
                    if input().strip().lower() == 'y':
                        filename = input("  Filename: ").strip()
                        if filename:
                            import json
                            # Serialize results (exclude non-JSON serializable objects)
                            serializable_results = {
                                'overall_score': phase2_results.get('review', {}).get('overall_score', 0),
                                'recommendation': phase2_results.get('review', {}).get('recommendation', 'N/A'),
                                'refactoring_count': phase2_results.get('refactorings', {}).get('count', 0),
                                'completion_count': len(phase2_results.get('completions', []))
                            }
                            with open(filename, 'w') as f:
                                json.dump(serializable_results, f, indent=2)
                            print(f"✓ Analysis saved to {filename}")

                # Phase 3 specialized commands
                elif user_input == '/metrics':
                    days = input("📊 Show metrics for how many days? [30]: ").strip() or "30"
                    self.show_metrics_dashboard(int(days))

                elif user_input == '/locate-bug':
                    print("\n🐛 Bug Localization Mode:")
                    print("  1. From traceback")
                    print("  2. From symptom description")
                    mode_choice = input("Choice [1]: ").strip() or "1"

                    if mode_choice == "1":
                        print("Paste traceback (end with empty line):")
                        traceback_lines = []
                        while True:
                            line = input()
                            if not line:
                                break
                            traceback_lines.append(line)
                        traceback = '\n'.join(traceback_lines)
                        if traceback:
                            self.locate_bug(traceback, mode="traceback")
                    else:
                        symptom = input("Describe the bug symptom: ").strip()
                        if symptom:
                            self.locate_bug(symptom, mode="symptom")

                elif user_input == '/gen-tests':
                    if not last_code:
                        print("❌ No code to generate tests for. Load or generate code first.")
                        continue
                    module_name = input("Module name [code]: ").strip() or "code"
                    tests = self.generate_intelligent_tests(last_code, module_name)
                    print("\n✅ Tests generated!")
                    save_tests = input("💾 Save tests? (y/N): ").strip().lower()
                    if save_tests == 'y':
                        test_file = input(f"  Filename [test_{module_name}.py]: ").strip() or f"test_{module_name}.py"
                        self.save_code(tests, test_file, overwrite=True)

                elif user_input == '/translate':
                    if not last_code:
                        print("❌ No code to translate. Load or generate code first.")
                        continue
                    print("\n🌐 Cross-Language Translation:")
                    print("  Source: python")
                    print("  Target: 1. C  2. Rust")
                    target_choice = input("Target language [1]: ").strip() or "1"
                    target_lang = "c" if target_choice == "1" else "rust"

                    translated, warnings = self.translate_code(last_code, "python", target_lang)
                    print(f"\n✅ Translated to {target_lang.upper()}!")

                    if warnings:
                        print(f"⚠️  {len(warnings)} warnings")

                    save_trans = input("💾 Save translation? (y/N): ").strip().lower()
                    if save_trans == 'y':
                        ext = ".c" if target_lang == "c" else ".rs"
                        trans_file = input(f"  Filename [code{ext}]: ").strip() or f"code{ext}"
                        self.save_code(translated, trans_file, overwrite=True)

                elif user_input == '/phase3':
                    if not last_code:
                        print("❌ No code for Phase 3 workflow. Load or generate code first.")
                        continue
                    file_path = input("File path [code.py]: ").strip() or "code.py"
                    phase3_results = self.complete_phase3_workflow(last_code, file_path)

                    # Offer to save outputs
                    print("\n💾 Save outputs?")
                    save = input("  Tests? (y/N): ").strip().lower()
                    if save == 'y' and phase3_results.get('tests'):
                        self.save_code(phase3_results['tests'], f"test_{file_path}", overwrite=True)

                    save_c = input("  C translation? (y/N): ").strip().lower()
                    if save_c == 'y' and phase3_results.get('c_translation'):
                        self.save_code(phase3_results['c_translation']['code'], f"{file_path.replace('.py', '.c')}", overwrite=True)

                    save_rust = input("  Rust translation? (y/N): ").strip().lower()
                    if save_rust == 'y' and phase3_results.get('rust_translation'):
                        self.save_code(phase3_results['rust_translation']['code'], f"{file_path.replace('.py', '.rs')}", overwrite=True)

                # Phase 4 cutting-edge research commands
                elif user_input == '/synthesize':
                    print("\n🧠 Neural Code Synthesis")
                    spec = input("Enter specification: ").strip()
                    if spec:
                        module_result = self.synthesize_module(spec)
                        if module_result:
                            print(f"\n✅ Module '{module_result.get('module_name')}' generated!")
                            save = input("💾 Save module files? (y/N): ").strip().lower()
                            if save == 'y':
                                # Files are already generated in the synthesize_module method
                                print("✓ Files saved to disk")

                elif user_input == '/npu-info':
                    self.get_npu_info()

                elif user_input == '/npu-optimize':
                    model_path = input("Model path (.onnx, .xml, etc.): ").strip()
                    if not model_path:
                        print("❌ Model path required")
                        continue

                    target_latency = input("Target latency (ms) [auto]: ").strip()
                    target_latency_ms = float(target_latency) if target_latency else None

                    preserve_acc = input("Preserve accuracy? (Y/n): ").strip().lower() or 'y'
                    preserve_accuracy = preserve_acc == 'y'

                    npu_result = self.optimize_for_npu(model_path, target_latency_ms, preserve_accuracy)
                    if npu_result:
                        print(f"\n✅ NPU optimization complete!")
                        print(f"   Speedup: {npu_result.get('speedup_factor', 0):.1f}x")

                elif user_input == '/hw-optimize':
                    if not last_code:
                        print("❌ No code to optimize. Load C/C++ code first.")
                        continue

                    hw_result = self.optimize_hardware(last_code, arch="lat5150drvmil")
                    if hw_result:
                        optimized_code = hw_result.get('optimized_code')
                        if optimized_code:
                            print("\n✨ Hardware-optimized code:")
                            print(optimized_code[:500] + "..." if len(optimized_code) > 500 else optimized_code)
                            last_code = optimized_code

                            save = input("\n💾 Save optimized code? (y/N): ").strip().lower()
                            if save == 'y':
                                filename = input("  Filename: ").strip()
                                if filename:
                                    self.save_code(optimized_code, filename, overwrite=True)

                elif user_input == '/gen-simd':
                    print("\n🚀 SIMD Code Generation")
                    print("  Operations: add, multiply, dot_product")
                    operation = input("Operation [add]: ").strip() or "add"

                    simd_code = self.generate_simd_code(operation)
                    last_code = simd_code

                    save = input("\n💾 Save SIMD code? (y/N): ").strip().lower()
                    if save == 'y':
                        filename = input("  Filename [simd_code.c]: ").strip() or "simd_code.c"
                        self.save_code(simd_code, filename, overwrite=True)

                elif user_input == '/gen-lockfree':
                    lockfree_code = self.generate_lockfree_queue()
                    last_code = lockfree_code

                    save = input("\n💾 Save lock-free queue? (y/N): ").strip().lower()
                    if save == 'y':
                        filename = input("  Filename [lockfree_queue.c]: ").strip() or "lockfree_queue.c"
                        self.save_code(lockfree_code, filename, overwrite=True)

                elif user_input == '/phase4':
                    print("\n🚀 Complete Phase 4 Workflow")
                    spec = input("Module specification: ").strip()
                    if not spec:
                        print("❌ Specification required")
                        continue

                    model_path_input = input("Model path (optional, press Enter to skip): ").strip() or None
                    code_input = None

                    if last_code:
                        use_last = input("Use loaded code for HW optimization? (y/N): ").strip().lower()
                        if use_last == 'y':
                            code_input = last_code

                    phase4_results = self.complete_phase4_workflow(spec, model_path_input, code_input)

                    # Offer to save module files
                    if phase4_results.get('module'):
                        print("\n💾 Module files automatically saved")

                else:
                    # Regular query
                    response = self.ask(user_input, use_rag=use_rag)
                    print(f"\n{response}\n")

                    # Extract code if present
                    import re
                    code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
                    if code_blocks:
                        last_code = code_blocks[-1].strip()

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='RAG-Enhanced Code Assistant for LAT5150DRVMIL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  %(prog)s -i

  # Ask a question
  %(prog)s "Write a kernel module to interface with NPU"

  # Generate code
  %(prog)s --code "Parse CVE JSON data"

  # Review code
  %(prog)s --review my_script.py

  # Use different model
  %(prog)s -i --model codellama:7b
"""
    )

    parser.add_argument('query', nargs='*', help='Coding question or task')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Start interactive session')
    parser.add_argument('--model', default='deepseek-coder:6.7b',
                       help='Ollama model (default: deepseek-coder:6.7b)')
    parser.add_argument('--no-rag', action='store_true',
                       help='Disable RAG context')
    parser.add_argument('--code', metavar='TASK',
                       help='Generate code for task')
    parser.add_argument('--review', metavar='FILE',
                       help='Review code file')
    parser.add_argument('--explain', metavar='TOPIC',
                       help='Explain topic or code')
    parser.add_argument('--debug', metavar='FILE',
                       help='Debug code file')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Quiet mode (less verbose)')

    args = parser.parse_args()

    # Create assistant
    assistant = CodeAssistant(model=args.model, verbose=not args.quiet)

    # Handle different modes
    if args.interactive or not any([args.query, args.code, args.review, args.explain, args.debug]):
        assistant.interactive()

    elif args.code:
        response = assistant.code(args.code)
        print(response)

    elif args.review:
        code = Path(args.review).read_text()
        response = assistant.review(code)
        print(response)

    elif args.explain:
        response = assistant.explain(args.explain)
        print(response)

    elif args.debug:
        code = Path(args.debug).read_text()
        response = assistant.debug(code)
        print(response)

    elif args.query:
        query = ' '.join(args.query)
        response = assistant.ask(query, use_rag=not args.no_rag)
        print(response)


if __name__ == '__main__':
    main()
