#!/usr/bin/env python3
"""
Dynamic Schema Generator
Based on ai-that-works Episode #25: "Dynamic Schema Generation"

Key Capabilities:
- LLM-driven Pydantic model generation from examples
- Natural language to structured schema
- Runtime model creation and validation
- Type inference and coercion
- Integration with AI engine

Benefits:
- Rapid prototyping without manual schema writing
- Adapt to changing data formats
- User-defined data structures
- Dynamic API response parsing
"""

import json
import re
from typing import Dict, Any, List, Optional, Type, get_type_hints
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

try:
    from pydantic import BaseModel, Field, create_model, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object


class SchemaComplexity(Enum):
    """Schema complexity levels"""
    SIMPLE = "simple"  # Flat dict with primitive types
    NESTED = "nested"  # Nested objects
    COMPLEX = "complex"  # Lists, unions, optional fields


@dataclass
class SchemaGenerationResult:
    """
    Result from schema generation

    Attributes:
        model: Generated Pydantic model class
        model_name: Name of the model
        schema_dict: JSON schema dictionary
        example_instance: Example instance of the model
        validation_passed: Whether validation passed
        generation_prompt: Prompt used for generation
        complexity: Schema complexity level
    """
    model: Optional[Type[BaseModel]]
    model_name: str
    schema_dict: Dict[str, Any]
    example_instance: Optional[Any] = None
    validation_passed: bool = False
    generation_prompt: str = ""
    complexity: SchemaComplexity = SchemaComplexity.SIMPLE
    error: Optional[str] = None


class DynamicSchemaGenerator:
    """
    Generate Pydantic models dynamically using LLM

    Use Cases:
    - Parse unstructured LLM outputs into structured data
    - Adapt to changing API response formats
    - User-defined data structures without manual coding
    - Rapid prototyping of data models
    - Type-safe dynamic configuration
    """

    def __init__(self, llm_engine=None):
        """
        Initialize dynamic schema generator

        Args:
            llm_engine: Optional LLM engine for schema generation
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required. Install: pip install pydantic")

        self.llm_engine = llm_engine
        self.generated_models: Dict[str, Type[BaseModel]] = {}
        self.generation_history: List[SchemaGenerationResult] = []

    def generate_from_examples(
        self,
        examples: List[Dict[str, Any]],
        model_name: str = "DynamicModel",
        description: Optional[str] = None
    ) -> SchemaGenerationResult:
        """
        Generate Pydantic model from example data

        Args:
            examples: List of example dictionaries
            model_name: Name for the generated model
            description: Optional model description

        Returns:
            SchemaGenerationResult with generated model
        """
        if not examples:
            return SchemaGenerationResult(
                model=None,
                model_name=model_name,
                schema_dict={},
                error="No examples provided"
            )

        # Infer schema from examples
        schema = self._infer_schema_from_examples(examples)

        # Determine complexity
        complexity = self._determine_complexity(schema)

        # Build prompt for LLM-based refinement (if available)
        prompt = self._build_schema_prompt(examples, model_name, description)

        # Try LLM-based generation if engine available
        if self.llm_engine:
            try:
                llm_schema = self._generate_schema_with_llm(prompt, examples)
                if llm_schema:
                    schema = llm_schema
            except Exception as e:
                print(f"⚠️  LLM schema generation failed, using heuristic: {e}")

        # Create Pydantic model
        try:
            model = self._create_pydantic_model(model_name, schema, description)

            # Validate with first example
            example_instance = None
            validation_passed = False
            try:
                example_instance = model(**examples[0])
                validation_passed = True
            except ValidationError as e:
                print(f"⚠️  Validation failed: {e}")

            result = SchemaGenerationResult(
                model=model,
                model_name=model_name,
                schema_dict=schema,
                example_instance=example_instance,
                validation_passed=validation_passed,
                generation_prompt=prompt,
                complexity=complexity
            )

            # Store model
            self.generated_models[model_name] = model
            self.generation_history.append(result)

            return result

        except Exception as e:
            return SchemaGenerationResult(
                model=None,
                model_name=model_name,
                schema_dict=schema,
                error=str(e),
                generation_prompt=prompt,
                complexity=complexity
            )

    def generate_from_natural_language(
        self,
        description: str,
        model_name: str = "DynamicModel"
    ) -> SchemaGenerationResult:
        """
        Generate Pydantic model from natural language description

        Args:
            description: Natural language description of the schema
            model_name: Name for the generated model

        Returns:
            SchemaGenerationResult with generated model
        """
        if not self.llm_engine:
            return SchemaGenerationResult(
                model=None,
                model_name=model_name,
                schema_dict={},
                error="LLM engine required for natural language generation"
            )

        # Build prompt for LLM
        prompt = f"""Generate a Pydantic model schema from this description:

{description}

Return a JSON schema with field names and types. Format:
{{
    "field_name": {{"type": "str", "description": "...", "required": true}},
    ...
}}

Supported types: str, int, float, bool, list, dict"""

        try:
            # Get LLM response
            response = self._query_llm(prompt)

            # Parse schema from response
            schema = self._parse_schema_from_llm(response)

            if not schema:
                return SchemaGenerationResult(
                    model=None,
                    model_name=model_name,
                    schema_dict={},
                    error="Failed to parse schema from LLM response"
                )

            # Create model
            model = self._create_pydantic_model(model_name, schema, description)

            complexity = self._determine_complexity(schema)

            result = SchemaGenerationResult(
                model=model,
                model_name=model_name,
                schema_dict=schema,
                validation_passed=True,
                generation_prompt=prompt,
                complexity=complexity
            )

            self.generated_models[model_name] = model
            self.generation_history.append(result)

            return result

        except Exception as e:
            return SchemaGenerationResult(
                model=None,
                model_name=model_name,
                schema_dict={},
                error=str(e),
                generation_prompt=prompt
            )

    def validate_data(
        self,
        model_name: str,
        data: Dict[str, Any]
    ) -> tuple[bool, Optional[Any], Optional[str]]:
        """
        Validate data against a generated model

        Args:
            model_name: Name of the model to validate against
            data: Data to validate

        Returns:
            Tuple of (success, validated_instance, error_message)
        """
        if model_name not in self.generated_models:
            return False, None, f"Model '{model_name}' not found"

        model = self.generated_models[model_name]

        try:
            instance = model(**data)
            return True, instance, None
        except ValidationError as e:
            return False, None, str(e)

    def _infer_schema_from_examples(
        self,
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Infer schema from example data using heuristics

        Args:
            examples: List of example dictionaries

        Returns:
            Schema dictionary
        """
        schema = {}

        # Analyze all examples to infer types
        all_keys = set()
        for example in examples:
            all_keys.update(example.keys())

        for key in all_keys:
            # Collect all values for this key
            values = [ex.get(key) for ex in examples if key in ex]

            # Infer type from values
            python_type = self._infer_type(values)

            # Check if required (present in all examples)
            required = all(key in ex for ex in examples)

            schema[key] = {
                "type": python_type,
                "required": required,
                "description": f"Auto-generated field for {key}"
            }

        return schema

    def _infer_type(self, values: List[Any]) -> str:
        """
        Infer type from list of values

        Args:
            values: List of values

        Returns:
            Type name as string
        """
        if not values:
            return "str"

        # Filter out None values for type inference
        non_none_values = [v for v in values if v is not None]

        if not non_none_values:
            return "Optional[str]"

        # Check for consistent types
        types = set(type(v).__name__ for v in non_none_values)

        if len(types) == 1:
            type_name = types.pop()

            # Map Python types to schema types
            type_map = {
                "str": "str",
                "int": "int",
                "float": "float",
                "bool": "bool",
                "list": "List[Any]",
                "dict": "Dict[str, Any]"
            }

            base_type = type_map.get(type_name, "str")

            # Add Optional if None values present
            if len(non_none_values) < len(values):
                return f"Optional[{base_type}]"

            return base_type

        # Mixed types - default to str
        return "str"

    def _determine_complexity(self, schema: Dict[str, Any]) -> SchemaComplexity:
        """Determine schema complexity"""
        has_nested = any(
            "Dict" in field.get("type", "") or "List" in field.get("type", "")
            for field in schema.values()
        )

        has_optional = any(
            "Optional" in field.get("type", "")
            for field in schema.values()
        )

        if has_nested or (has_optional and len(schema) > 5):
            return SchemaComplexity.COMPLEX
        elif has_optional or len(schema) > 3:
            return SchemaComplexity.NESTED
        else:
            return SchemaComplexity.SIMPLE

    def _create_pydantic_model(
        self,
        model_name: str,
        schema: Dict[str, Any],
        description: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Create Pydantic model from schema

        Args:
            model_name: Name for the model
            schema: Schema dictionary
            description: Optional model description

        Returns:
            Pydantic model class
        """
        # Build field definitions
        fields = {}

        for field_name, field_info in schema.items():
            field_type = self._convert_type_string(field_info["type"])
            required = field_info.get("required", True)
            field_desc = field_info.get("description", "")

            if required:
                fields[field_name] = (field_type, Field(..., description=field_desc))
            else:
                fields[field_name] = (field_type, Field(None, description=field_desc))

        # Create model
        model = create_model(
            model_name,
            __doc__=description,
            **fields
        )

        return model

    def _convert_type_string(self, type_str: str) -> Type:
        """
        Convert type string to Python type

        Args:
            type_str: Type as string (e.g., "str", "Optional[int]")

        Returns:
            Python type
        """
        # Handle Optional types
        optional_match = re.match(r"Optional\[(.*)\]", type_str)
        if optional_match:
            inner_type = self._convert_type_string(optional_match.group(1))
            return Optional[inner_type]

        # Handle List types
        list_match = re.match(r"List\[(.*)\]", type_str)
        if list_match:
            inner_type = self._convert_type_string(list_match.group(1))
            return List[inner_type]

        # Handle Dict types
        dict_match = re.match(r"Dict\[(.*),\s*(.*)\]", type_str)
        if dict_match:
            key_type = self._convert_type_string(dict_match.group(1))
            value_type = self._convert_type_string(dict_match.group(2))
            return Dict[key_type, value_type]

        # Basic types
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "Any": Any
        }

        return type_map.get(type_str, str)

    def _build_schema_prompt(
        self,
        examples: List[Dict],
        model_name: str,
        description: Optional[str]
    ) -> str:
        """Build prompt for LLM-based schema generation"""
        prompt = f"""Analyze these examples and generate a JSON schema:

Examples:
{json.dumps(examples[:3], indent=2)}

Model name: {model_name}
"""

        if description:
            prompt += f"Description: {description}\n"

        prompt += """
Generate a schema with field names, types, and whether required.
Return JSON format:
{
    "field_name": {"type": "str", "required": true, "description": "..."},
    ...
}"""

        return prompt

    def _generate_schema_with_llm(
        self,
        prompt: str,
        examples: List[Dict]
    ) -> Optional[Dict]:
        """Generate schema using LLM"""
        try:
            response = self._query_llm(prompt)
            return self._parse_schema_from_llm(response)
        except Exception as e:
            print(f"⚠️  LLM generation failed: {e}")
            return None

    def _query_llm(self, prompt: str) -> str:
        """Query LLM engine"""
        if hasattr(self.llm_engine, 'query'):
            result = self.llm_engine.query(
                prompt,
                use_rag=False,
                use_cache=False
            )
            return result.content if hasattr(result, 'content') else str(result)
        elif hasattr(self.llm_engine, 'generate'):
            result = self.llm_engine.generate(prompt)
            return result.get('response', '')
        else:
            raise ValueError("LLM engine does not have query() or generate() method")

    def _parse_schema_from_llm(self, response: str) -> Optional[Dict]:
        """Parse schema from LLM response"""
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    def get_model(self, model_name: str) -> Optional[Type[BaseModel]]:
        """Get a generated model by name"""
        return self.generated_models.get(model_name)

    def list_models(self) -> List[str]:
        """List all generated model names"""
        return list(self.generated_models.keys())

    def export_schema(
        self,
        model_name: str,
        output_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Export model schema to JSON

        Args:
            model_name: Name of model to export
            output_path: Optional file path to save

        Returns:
            JSON schema string or None if model not found
        """
        if model_name not in self.generated_models:
            return None

        # Find schema in history
        schema_dict = None
        for result in self.generation_history:
            if result.model_name == model_name:
                schema_dict = result.schema_dict
                break

        if not schema_dict:
            return None

        schema_json = json.dumps(schema_dict, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(schema_json)

        return schema_json

    def get_statistics(self) -> Dict[str, Any]:
        """Get schema generator statistics"""
        return {
            "models_generated": len(self.generated_models),
            "generation_history": len(self.generation_history),
            "models": list(self.generated_models.keys()),
            "complexity_breakdown": {
                "simple": sum(1 for r in self.generation_history if r.complexity == SchemaComplexity.SIMPLE),
                "nested": sum(1 for r in self.generation_history if r.complexity == SchemaComplexity.NESTED),
                "complex": sum(1 for r in self.generation_history if r.complexity == SchemaComplexity.COMPLEX)
            },
            "validation_success_rate": sum(1 for r in self.generation_history if r.validation_passed) / max(len(self.generation_history), 1)
        }


def main():
    """Demo usage"""
    print("=== Dynamic Schema Generator Demo ===\n")

    generator = DynamicSchemaGenerator()

    # Example 1: Generate from examples
    print("1. Generate schema from examples:")
    examples = [
        {"name": "Alice", "age": 30, "email": "alice@example.com", "verified": True},
        {"name": "Bob", "age": 25, "email": "bob@example.com", "verified": False},
        {"name": "Charlie", "age": 35, "email": "charlie@example.com"}
    ]

    result = generator.generate_from_examples(
        examples,
        model_name="User",
        description="User model with basic information"
    )

    if result.model:
        print(f"   ✅ Model '{result.model_name}' generated")
        print(f"   Complexity: {result.complexity.value}")
        print(f"   Validation: {'✅' if result.validation_passed else '❌'}")
        print(f"   Schema: {json.dumps(result.schema_dict, indent=6)}")

        # Validate new data
        print("\n2. Validate new data:")
        new_data = {"name": "David", "age": 28, "email": "david@example.com", "verified": True}
        success, instance, error = generator.validate_data("User", new_data)

        if success:
            print(f"   ✅ Valid: {instance}")
        else:
            print(f"   ❌ Invalid: {error}")

    # Statistics
    print("\n3. Statistics:")
    stats = generator.get_statistics()
    print(f"   Models generated: {stats['models_generated']}")
    print(f"   Complexity breakdown: {stats['complexity_breakdown']}")
    print(f"   Validation success rate: {stats['validation_success_rate']:.1%}")


if __name__ == "__main__":
    main()
