#!/usr/bin/env python3
"""
Code Specialist - Auto-Coding with Hardware Optimization

Uses specialized coding models with maximum hardware utilization:
- DeepSeek Coder 6.7B: Fast, code-specialized
- Qwen 2.5 Coder 7B: High quality code generation
- Code Llama 13B/70B: Code review and complex tasks

Auto-routing based on task complexity.
"""

import sys
import os
import re
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dsmil_ai_engine import DSMILAIEngine

class CodeSpecialist:
    def __init__(self):
        self.engine = DSMILAIEngine()

        # Specialized coding models (NOW AVAILABLE!)
        self.code_models = {
            "fast_code": "deepseek-coder:6.7b-instruct",     # Quick snippets, functions ✅ Downloaded
            "quality_code": "qwen2.5-coder:7b",               # Complex implementations ✅ Downloaded
            "review": "codellama:70b",                        # Code review (use 70B, no 13B)
            "large_review": "codellama:70b"                   # Deep analysis ✅ Available
        }

        # Code task patterns
        self.code_patterns = {
            "function": r"(write|create|implement|build|make).*(function|method|def)",
            "class": r"(write|create|implement|build|make).*(class|object|module)",
            "script": r"(write|create|implement|build|make).*(script|program|tool)",
            "refactor": r"(refactor|improve|optimize|clean|rewrite)",
            "debug": r"(debug|fix|solve|find bug|error)",
            "review": r"(review|analyze|check|audit|assess).*(code|implementation)",
            "explain": r"(explain|describe|how does|what does|understand)",
        }

    def detect_code_task(self, query):
        """
        Detect if query is code-related and classify complexity

        Returns: (is_code, task_type, complexity)
        """
        query_lower = query.lower()

        # Check for code keywords
        code_keywords = [
            'code', 'function', 'class', 'implement', 'python', 'script',
            'javascript', 'java', 'c++', 'rust', 'go', 'refactor',
            'debug', 'algorithm', 'api', 'library', 'module'
        ]

        is_code = any(keyword in query_lower for keyword in code_keywords)

        if not is_code:
            return False, None, None

        # Classify task type
        task_type = "general_code"
        for pattern_name, pattern in self.code_patterns.items():
            if re.search(pattern, query_lower):
                task_type = pattern_name
                break

        # Determine complexity
        complexity_indicators = {
            "simple": ["snippet", "example", "simple", "quick", "small"],
            "medium": ["function", "method", "class", "module"],
            "complex": ["system", "architecture", "framework", "large", "complete", "full"]
        }

        complexity = "medium"  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
                break

        # Override: review/explain are always fast tasks
        if task_type in ["review", "explain"]:
            complexity = "simple"

        return True, task_type, complexity

    def select_model(self, task_type, complexity):
        """Select best model for code task"""

        # Review tasks → specialized review models
        if task_type == "review":
            if complexity == "complex":
                return self.code_models["large_review"]
            return self.code_models["review"]

        # Explain tasks → fast model is fine
        if task_type == "explain":
            return self.code_models["fast_code"]

        # Code generation by complexity
        if complexity == "simple":
            return self.code_models["fast_code"]      # DeepSeek 6.7B
        elif complexity == "complex":
            return self.code_models["quality_code"]   # Qwen 7B
        else:
            return self.code_models["fast_code"]      # Default to fast

    def generate_code(self, query, model_override=None):
        """
        Generate code with automatic model selection

        Args:
            query: Code task description
            model_override: Force specific model

        Returns:
            Code generation result with DSMIL attestation
        """
        start_time = time.time()

        # Detect code task
        is_code, task_type, complexity = self.detect_code_task(query)

        if not is_code:
            return {
                "error": "Not a code task",
                "suggestion": "Use regular AI query for non-code questions"
            }

        # Select model
        if model_override:
            selected_model = model_override
        else:
            selected_model = self.select_model(task_type, complexity)

        # Check if model available
        if not self.engine.check_model_available(selected_model):
            # Fallback chain
            fallbacks = [
                self.code_models["fast_code"],
                self.engine.models["fast"],  # DeepSeek R1
                self.code_models["quality_code"]
            ]

            selected_model = None
            for fallback in fallbacks:
                if self.engine.check_model_available(fallback):
                    selected_model = fallback
                    break

            if not selected_model:
                return {
                    "error": "No coding models available",
                    "suggestion": f"Download: ollama pull {self.code_models['fast_code']}"
                }

        # Build code-optimized prompt
        code_prompt = f"""You are an expert programmer. Generate clean, efficient, well-documented code.

Task: {query}

Requirements:
- Include docstrings/comments
- Handle edge cases
- Follow best practices
- Make it production-ready

Code:"""

        # Generate
        result = self.engine.generate(code_prompt, model_selection=selected_model)

        # Add code metadata
        result['code_task'] = {
            "task_type": task_type,
            "complexity": complexity,
            "model_selected": selected_model,
            "total_time": round(time.time() - start_time, 2)
        }

        return result

    def review_code(self, code, focus=None):
        """Review existing code for issues"""

        review_prompt = f"""Review this code for issues, bugs, and improvements.

{'Focus on: ' + focus if focus else 'Comprehensive review'}

Code:
```
{code}
```

Provide:
1. Issues/bugs found
2. Security concerns
3. Performance improvements
4. Best practice violations
5. Suggested fixes"""

        return self.engine.generate(review_prompt, model_selection=self.code_models["review"])

# CLI
if __name__ == "__main__":
    import json

    specialist = CodeSpecialist()

    if len(sys.argv) < 2:
        print("Code Specialist - Usage:")
        print("  python3 code_specialist.py generate 'write a function to...'")
        print("  python3 code_specialist.py review 'code here' [focus]")
        print("  python3 code_specialist.py detect 'query'")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "generate" and len(sys.argv) > 2:
        query = sys.argv[2]
        print(f"\n🎯 Code Task: {query}\n")

        result = specialist.generate_code(query)

        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            if 'suggestion' in result:
                print(f"💡 {result['suggestion']}")
        else:
            print("=" * 60)
            print(result['response'])
            print("=" * 60)
            print(f"\nTask: {result['code_task']['task_type']} ({result['code_task']['complexity']})")
            print(f"Model: {result.get('model', 'N/A')}")
            print(f"Time: {result.get('inference_time', 'N/A')}s")
            print(f"DSMIL Verified: {result.get('attestation', {}).get('verified', False)}")

    elif cmd == "detect" and len(sys.argv) > 2:
        query = sys.argv[2]
        is_code, task_type, complexity = specialist.detect_code_task(query)
        print(json.dumps({
            "is_code_task": is_code,
            "task_type": task_type,
            "complexity": complexity,
            "suggested_model": specialist.select_model(task_type, complexity) if is_code else None
        }, indent=2))

    elif cmd == "review" and len(sys.argv) > 2:
        code = sys.argv[2]
        focus = sys.argv[3] if len(sys.argv) > 3 else None
        result = specialist.review_code(code, focus)
        print(result.get('response', 'Error'))
