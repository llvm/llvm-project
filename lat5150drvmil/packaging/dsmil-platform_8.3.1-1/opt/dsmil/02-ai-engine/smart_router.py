#!/usr/bin/env python3
"""
Smart Router - Intelligent Query Routing System

Automatically routes queries to the best model:
- Code tasks → DeepSeek Coder or Qwen Coder
- General queries → DeepSeek R1
- Complex analysis → CodeLlama 70B
- Multimodal → Gemini Pro
- Web search needed → Add web search first

100% automatic, invisible to user. Just type and get optimal results.
"""

import re
from typing import Tuple, Optional

class SmartRouter:
    def __init__(self):
        """Initialize smart router with detection patterns"""

        # Code detection patterns (comprehensive)
        self.code_keywords = {
            'actions': ['write', 'create', 'implement', 'build', 'make', 'generate', 'develop', 'code'],
            'artifacts': ['function', 'class', 'method', 'module', 'script', 'program', 'api', 'endpoint', 'service'],
            'languages': ['python', 'javascript', 'typescript', 'java', 'c++', 'rust', 'go', 'php', 'ruby', 'swift'],
            'operations': ['debug', 'fix', 'refactor', 'optimize', 'test', 'deploy', 'compile'],
            'concepts': ['algorithm', 'data structure', 'regex', 'sql', 'http', 'rest', 'graphql', 'async']
        }

        # Web search detection patterns
        self.web_search_keywords = {
            'temporal': ['latest', 'recent', 'today', 'this week', 'current', 'now', 'update'],
            'questions': ['what happened', 'who is', 'when did', 'where is', 'news about'],
            'research': ['papers on', 'research about', 'find information', 'search for']
        }

        # Complexity indicators
        self.complex_indicators = ['comprehensive', 'detailed analysis', 'research', 'investigate', 'explore', 'compare multiple']

    def detect_code_task(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Detect if query is code-related

        Returns:
            (is_code, task_type, complexity)
            task_type: 'function', 'class', 'script', 'refactor', 'debug', 'explain', None
            complexity: 'simple', 'medium', 'complex', None
        """
        query_lower = query.lower()

        # Check for code keywords
        action_match = any(keyword in query_lower for keyword in self.code_keywords['actions'])
        artifact_match = any(keyword in query_lower for keyword in self.code_keywords['artifacts'])
        language_match = any(keyword in query_lower for keyword in self.code_keywords['languages'])
        operation_match = any(keyword in query_lower for keyword in self.code_keywords['operations'])
        concept_match = any(keyword in query_lower for keyword in self.code_keywords['concepts'])

        # Code if: (action + artifact) OR language OR (operation + programming context)
        is_code = (action_match and artifact_match) or language_match or operation_match or concept_match

        if not is_code:
            return False, None, None

        # Determine task type
        task_type = 'general_code'
        if any(word in query_lower for word in ['function', 'method', 'def ']):
            task_type = 'function'
        elif any(word in query_lower for word in ['class', 'object', 'struct']):
            task_type = 'class'
        elif any(word in query_lower for word in ['script', 'program', 'tool']):
            task_type = 'script'
        elif any(word in query_lower for word in ['refactor', 'improve', 'optimize']):
            task_type = 'refactor'
        elif any(word in query_lower for word in ['debug', 'fix', 'bug', 'error']):
            task_type = 'debug'
        elif any(word in query_lower for word in ['explain', 'what does', 'how does']):
            task_type = 'explain'

        # Determine complexity
        complexity = 'medium'  # default
        if any(word in query_lower for word in ['simple', 'basic', 'quick', 'small', 'snippet']):
            complexity = 'simple'
        elif any(word in query_lower for word in ['complex', 'system', 'architecture', 'framework', 'large']):
            complexity = 'complex'

        # Override: explain is always simple
        if task_type == 'explain':
            complexity = 'simple'

        return True, task_type, complexity

    def detect_web_search_needed(self, query: str) -> bool:
        """Detect if query needs web search"""
        query_lower = query.lower()

        # Check for temporal or research keywords
        temporal_match = any(keyword in query_lower for keyword in self.web_search_keywords['temporal'])
        question_match = any(keyword in query_lower for keyword in self.web_search_keywords['questions'])
        research_match = any(keyword in query_lower for keyword in self.web_search_keywords['research'])

        return temporal_match or question_match or research_match

    def route(self, query: str, has_images: bool = False, user_preference: str = None) -> dict:
        """
        Main routing function - decides which model to use

        Args:
            query: User's query
            has_images: Whether query includes images
            user_preference: Optional override ("code", "fast", "large", etc.)

        Returns:
            Dict with routing decision
        """

        # User override takes precedence
        if user_preference and user_preference != "auto":
            return {
                "model": user_preference,
                "reason": "user_preference",
                "explanation": f"User selected {user_preference}"
            }

        # Multimodal detection
        if has_images:
            return {
                "model": "gemini-pro",
                "reason": "multimodal",
                "explanation": "Images detected - Gemini required",
                "web_search": False
            }

        # Web search detection
        needs_web = self.detect_web_search_needed(query)

        # Code detection
        is_code, task_type, complexity = self.detect_code_task(query)

        if is_code:
            # Route to coding models based on complexity
            if complexity == 'simple' or task_type == 'explain':
                model = "deepseek-coder:6.7b-instruct"
                explanation = f"Code task detected: {task_type} ({complexity})"
            elif complexity == 'complex':
                model = "qwen2.5-coder:7b"
                explanation = f"Complex code task: {task_type}"
            else:
                model = "deepseek-coder:6.7b-instruct"
                explanation = f"Code task: {task_type}"

            return {
                "model": model,
                "reason": "code_detected",
                "task_type": task_type,
                "complexity": complexity,
                "explanation": explanation,
                "web_search": needs_web
            }

        # General queries
        query_length = len(query)
        word_count = len(query.split())

        # Check for complexity indicators
        is_complex = any(indicator in query.lower() for indicator in self.complex_indicators)

        if is_complex or query_length > 300 or word_count > 50:
            return {
                "model": "codellama:70b",
                "reason": "complex_query",
                "explanation": "Complex analysis detected",
                "web_search": needs_web
            }

        # Default: fast general model
        return {
            "model": "deepseek-r1:1.5b",
            "reason": "general_query",
            "explanation": "General question",
            "web_search": needs_web
        }

    def explain_routing(self, routing_decision: dict, format: str = "text") -> str:
        """
        Generate human-readable explanation of routing decision

        Args:
            routing_decision: Dict from route()
            format: "text", "emoji", "short"

        Returns:
            Formatted explanation string
        """
        model = routing_decision['model']
        reason = routing_decision.get('explanation', routing_decision['reason'])

        if format == "emoji":
            emoji_map = {
                'code_detected': '💻',
                'complex_query': '🧠',
                'general_query': '💬',
                'multimodal': '🖼️',
                'user_preference': '👤'
            }
            emoji = emoji_map.get(routing_decision['reason'], '🤖')
            return f"{emoji} {model.split(':')[0]} | {reason}"

        elif format == "short":
            model_short = model.split(':')[0].replace('-', ' ').title()
            return f"{model_short} ({reason})"

        else:  # text
            return f"Using {model} - {reason}"

# CLI
if __name__ == "__main__":
    import sys
    import json

    router = SmartRouter()

    if len(sys.argv) < 2:
        print("Smart Router - Usage:")
        print("  python3 smart_router.py 'write a python function'")
        print("  python3 smart_router.py 'what is quantum computing'")
        print("  python3 smart_router.py 'latest news about AI'")
        sys.exit(1)

    query = ' '.join(sys.argv[1:])
    decision = router.route(query)

    print(json.dumps(decision, indent=2))
    print(f"\n{router.explain_routing(decision, format='emoji')}")
