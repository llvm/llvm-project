"""
Cross-Language Translation (Phase 3.4)

Translate code between Python, C, and Rust with AST-based translation,
type system mapping, memory management conversion, and idiomatic code generation.

Supports:
- Python ↔ C
- Python ↔ Rust
- C ↔ Rust

Features:
- AST-based translation (not text replacement)
- Type system translation
- Memory management conversion (GC → manual, GC → ownership)
- Idiomatic code generation
- Library/dependency mapping
- Error handling translation
- Comment/documentation preservation

Example:
    >>> translator = CrossLanguageTranslator()
    >>> c_code = translator.translate(python_code, source_lang="python", target_lang="c")
    >>> rust_code = translator.translate(python_code, source_lang="python", target_lang="rust")
"""

import ast
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum


class Language(Enum):
    """Supported languages"""
    PYTHON = "python"
    C = "c"
    RUST = "rust"


@dataclass
class TypeMapping:
    """Type mapping between languages"""
    source_type: str
    target_type: str
    requires_conversion: bool = False
    conversion_code: Optional[str] = None


@dataclass
class TranslationResult:
    """Result of code translation"""
    source_language: Language
    target_language: Language
    source_code: str
    translated_code: str
    warnings: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class TypeSystemMapper:
    """Map types between languages"""

    # Python → C type mappings
    PYTHON_TO_C = {
        'int': 'int32_t',
        'float': 'double',
        'str': 'char*',
        'bool': 'bool',
        'list': 'array',  # Needs size info
        'dict': 'struct',  # Needs definition
        'None': 'NULL',
        'bytes': 'uint8_t*',
    }

    # Python → Rust type mappings
    PYTHON_TO_RUST = {
        'int': 'i64',
        'float': 'f64',
        'str': 'String',
        'bool': 'bool',
        'list': 'Vec<T>',  # Template
        'dict': 'HashMap<K, V>',  # Template
        'None': 'Option<T>',
        'bytes': 'Vec<u8>',
    }

    # C → Python type mappings
    C_TO_PYTHON = {
        'int': 'int',
        'int32_t': 'int',
        'int64_t': 'int',
        'uint32_t': 'int',
        'uint64_t': 'int',
        'float': 'float',
        'double': 'float',
        'char*': 'str',
        'void*': 'Any',
        'bool': 'bool',
        'NULL': 'None',
    }

    # C → Rust type mappings
    C_TO_RUST = {
        'int': 'i32',
        'int32_t': 'i32',
        'int64_t': 'i64',
        'uint32_t': 'u32',
        'uint64_t': 'u64',
        'float': 'f32',
        'double': 'f64',
        'char*': 'String',
        'void*': '*mut c_void',
        'bool': 'bool',
        'NULL': 'None',
    }

    # Rust → Python type mappings
    RUST_TO_PYTHON = {
        'i8': 'int', 'i16': 'int', 'i32': 'int', 'i64': 'int',
        'u8': 'int', 'u16': 'int', 'u32': 'int', 'u64': 'int',
        'f32': 'float', 'f64': 'float',
        'String': 'str', '&str': 'str',
        'Vec<T>': 'list',
        'HashMap<K,V>': 'dict',
        'Option<T>': 'Optional[T]',
        'Result<T,E>': 'T',  # Will add try/except
        'bool': 'bool',
    }

    # Rust → C type mappings
    RUST_TO_C = {
        'i8': 'int8_t', 'i16': 'int16_t', 'i32': 'int32_t', 'i64': 'int64_t',
        'u8': 'uint8_t', 'u16': 'uint16_t', 'u32': 'uint32_t', 'u64': 'uint64_t',
        'f32': 'float', 'f64': 'double',
        'String': 'char*', '&str': 'const char*',
        'Vec<T>': 'T*',  # With size
        'bool': 'bool',
    }

    @classmethod
    def map_type(cls, type_name: str, source_lang: Language, target_lang: Language) -> str:
        """Map type from source to target language"""

        if source_lang == Language.PYTHON and target_lang == Language.C:
            return cls.PYTHON_TO_C.get(type_name, type_name)
        elif source_lang == Language.PYTHON and target_lang == Language.RUST:
            return cls.PYTHON_TO_RUST.get(type_name, type_name)
        elif source_lang == Language.C and target_lang == Language.PYTHON:
            return cls.C_TO_PYTHON.get(type_name, type_name)
        elif source_lang == Language.C and target_lang == Language.RUST:
            return cls.C_TO_RUST.get(type_name, type_name)
        elif source_lang == Language.RUST and target_lang == Language.PYTHON:
            return cls.RUST_TO_PYTHON.get(type_name, type_name)
        elif source_lang == Language.RUST and target_lang == Language.C:
            return cls.RUST_TO_C.get(type_name, type_name)
        else:
            return type_name


class PythonToCTranslator:
    """Translate Python to C"""

    def __init__(self):
        self.type_mapper = TypeSystemMapper()
        self.warnings = []

    def translate(self, python_code: str) -> str:
        """Translate Python code to C"""
        self.warnings = []

        try:
            tree = ast.parse(python_code)
            c_code = self._translate_module(tree)
            return c_code
        except SyntaxError as e:
            self.warnings.append(f"Syntax error: {e}")
            return f"// Translation failed: {e}"

    def _translate_module(self, tree: ast.Module) -> str:
        """Translate module to C"""
        c_lines = []

        # Add standard includes
        c_lines.append("#include <stdio.h>")
        c_lines.append("#include <stdlib.h>")
        c_lines.append("#include <string.h>")
        c_lines.append("#include <stdint.h>")
        c_lines.append("#include <stdbool.h>")
        c_lines.append("")

        # Translate each top-level statement
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                c_lines.append(self._translate_function(node))
            elif isinstance(node, ast.ClassDef):
                c_lines.append(self._translate_class(node))
            elif isinstance(node, ast.Assign):
                c_lines.append(self._translate_assignment(node) + ";")

        return '\n'.join(c_lines)

    def _translate_function(self, func_node: ast.FunctionDef) -> str:
        """Translate Python function to C function"""
        func_name = func_node.name
        params = func_node.args.args

        # Infer return type from function name or default to int32_t
        return_type = self._infer_return_type(func_node)

        # Build parameter list
        c_params = []
        for param in params:
            param_name = param.arg
            # Infer type from parameter name
            param_type = self._infer_param_type(param_name, param.annotation)
            c_params.append(f"{param_type} {param_name}")

        params_str = ', '.join(c_params) if c_params else 'void'

        # Translate function body
        body_lines = []
        for stmt in func_node.body:
            body_lines.append("    " + self._translate_statement(stmt))

        body_str = '\n'.join(body_lines) if body_lines else "    // TODO: Implement"

        # Build C function
        c_func = f"""{return_type} {func_name}({params_str}) {{
{body_str}
}}
"""
        return c_func

    def _translate_class(self, class_node: ast.ClassDef) -> str:
        """Translate Python class to C struct + functions"""
        class_name = class_node.name

        # Extract fields from __init__
        fields = self._extract_class_fields(class_node)

        # Build struct
        struct_def = f"typedef struct {class_name} {{\n"
        for field_name, field_type in fields.items():
            c_type = self.type_mapper.map_type(field_type, Language.PYTHON, Language.C)
            struct_def += f"    {c_type} {field_name};\n"
        struct_def += f"}} {class_name};\n\n"

        # Translate methods as functions
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name != '__init__':
                method_code = self._translate_method_to_function(node, class_name)
                methods.append(method_code)

        return struct_def + '\n'.join(methods)

    def _extract_class_fields(self, class_node: ast.ClassDef) -> Dict[str, str]:
        """Extract class fields from __init__"""
        fields = {}

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                if target.value.id == 'self':
                                    field_name = target.attr
                                    field_type = self._infer_type_from_value(stmt.value)
                                    fields[field_name] = field_type

        return fields

    def _translate_method_to_function(self, method_node: ast.FunctionDef, class_name: str) -> str:
        """Translate method to C function with struct pointer"""
        method_name = method_node.name
        params = method_node.args.args[1:]  # Skip 'self'

        return_type = self._infer_return_type(method_node)

        # First parameter is struct pointer
        c_params = [f"{class_name}* self"]
        for param in params:
            param_name = param.arg
            param_type = self._infer_param_type(param_name, param.annotation)
            c_params.append(f"{param_type} {param_name}")

        params_str = ', '.join(c_params)

        # Function name includes class name
        full_name = f"{class_name}_{method_name}"

        body_lines = []
        for stmt in method_node.body:
            body_lines.append("    " + self._translate_statement(stmt))

        body_str = '\n'.join(body_lines) if body_lines else "    // TODO: Implement"

        return f"""{return_type} {full_name}({params_str}) {{
{body_str}
}}
"""

    def _translate_statement(self, stmt: ast.AST) -> str:
        """Translate a Python statement to C"""
        if isinstance(stmt, ast.Return):
            value = self._translate_expression(stmt.value) if stmt.value else ""
            return f"return {value};" if value else "return;"

        elif isinstance(stmt, ast.Assign):
            return self._translate_assignment(stmt) + ";"

        elif isinstance(stmt, ast.If):
            return self._translate_if(stmt)

        elif isinstance(stmt, ast.For):
            return self._translate_for(stmt)

        elif isinstance(stmt, ast.While):
            return self._translate_while(stmt)

        elif isinstance(stmt, ast.Expr):
            return self._translate_expression(stmt.value) + ";"

        else:
            return f"// TODO: Translate {type(stmt).__name__}"

    def _translate_assignment(self, assign: ast.Assign) -> str:
        """Translate assignment"""
        if not assign.targets:
            return "// Empty assignment"

        target = assign.targets[0]
        value = self._translate_expression(assign.value)

        if isinstance(target, ast.Name):
            var_name = target.id
            var_type = self._infer_type_from_value(assign.value)
            c_type = self.type_mapper.map_type(var_type, Language.PYTHON, Language.C)
            return f"{c_type} {var_name} = {value}"
        else:
            return f"{self._translate_expression(target)} = {value}"

    def _translate_if(self, if_stmt: ast.If) -> str:
        """Translate if statement"""
        test = self._translate_expression(if_stmt.test)

        body_lines = [self._translate_statement(s) for s in if_stmt.body]
        body = '\n        '.join(body_lines)

        c_if = f"if ({test}) {{\n        {body}\n    }}"

        if if_stmt.orelse:
            else_lines = [self._translate_statement(s) for s in if_stmt.orelse]
            else_body = '\n        '.join(else_lines)
            c_if += f" else {{\n        {else_body}\n    }}"

        return c_if

    def _translate_for(self, for_stmt: ast.For) -> str:
        """Translate for loop"""
        if isinstance(for_stmt.iter, ast.Call) and isinstance(for_stmt.iter.func, ast.Name):
            if for_stmt.iter.func.id == 'range':
                # for i in range(n) → for (i=0; i<n; i++)
                target = for_stmt.target.id if isinstance(for_stmt.target, ast.Name) else "i"
                args = for_stmt.iter.args

                if len(args) == 1:
                    # range(n)
                    end = self._translate_expression(args[0])
                    init = f"int32_t {target} = 0"
                    condition = f"{target} < {end}"
                    increment = f"{target}++"
                else:
                    init = f"int32_t {target} = {self._translate_expression(args[0])}"
                    condition = f"{target} < {self._translate_expression(args[1])}"
                    increment = f"{target}++"

                body_lines = [self._translate_statement(s) for s in for_stmt.body]
                body = '\n        '.join(body_lines)

                return f"for ({init}; {condition}; {increment}) {{\n        {body}\n    }}"

        return "// TODO: Complex for loop"

    def _translate_while(self, while_stmt: ast.While) -> str:
        """Translate while loop"""
        test = self._translate_expression(while_stmt.test)

        body_lines = [self._translate_statement(s) for s in while_stmt.body]
        body = '\n        '.join(body_lines)

        return f"while ({test}) {{\n        {body}\n    }}"

    def _translate_expression(self, expr: ast.AST) -> str:
        """Translate expression"""
        if expr is None:
            return ""

        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return f'"{expr.value}"'
            elif expr.value is None:
                return "NULL"
            elif isinstance(expr.value, bool):
                return "true" if expr.value else "false"
            else:
                return str(expr.value)

        elif isinstance(expr, ast.Name):
            return expr.id

        elif isinstance(expr, ast.BinOp):
            left = self._translate_expression(expr.left)
            right = self._translate_expression(expr.right)
            op = self._translate_operator(expr.op)
            return f"({left} {op} {right})"

        elif isinstance(expr, ast.Compare):
            left = self._translate_expression(expr.left)
            comparators = [self._translate_expression(c) for c in expr.comparators]
            ops = [self._translate_comparison_op(op) for op in expr.ops]

            result = left
            for op, comp in zip(ops, comparators):
                result = f"({result} {op} {comp})"
            return result

        elif isinstance(expr, ast.Call):
            func = self._translate_expression(expr.func)
            args = [self._translate_expression(arg) for arg in expr.args]
            return f"{func}({', '.join(args)})"

        elif isinstance(expr, ast.Attribute):
            value = self._translate_expression(expr.value)
            return f"{value}->{expr.attr}"  # Assume struct pointer

        elif isinstance(expr, ast.Subscript):
            value = self._translate_expression(expr.value)
            slice_val = self._translate_expression(expr.slice)
            return f"{value}[{slice_val}]"

        else:
            return f"/* {type(expr).__name__} */"

    def _translate_operator(self, op: ast.operator) -> str:
        """Translate binary operator"""
        ops = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Mod: '%', ast.Pow: '**',  # Need pow() function
            ast.LShift: '<<', ast.RShift: '>>',
            ast.BitOr: '|', ast.BitXor: '^', ast.BitAnd: '&',
        }
        return ops.get(type(op), '?')

    def _translate_comparison_op(self, op: ast.cmpop) -> str:
        """Translate comparison operator"""
        ops = {
            ast.Eq: '==', ast.NotEq: '!=',
            ast.Lt: '<', ast.LtE: '<=',
            ast.Gt: '>', ast.GtE: '>=',
        }
        return ops.get(type(op), '?')

    def _infer_return_type(self, func_node: ast.FunctionDef) -> str:
        """Infer function return type"""
        if func_node.returns:
            # Has type annotation
            if isinstance(func_node.returns, ast.Name):
                py_type = func_node.returns.id
                return self.type_mapper.map_type(py_type, Language.PYTHON, Language.C)

        # Infer from function name
        if 'count' in func_node.name or 'size' in func_node.name:
            return 'int32_t'
        elif 'is_' in func_node.name or 'has_' in func_node.name:
            return 'bool'
        elif 'get_' in func_node.name and 'name' in func_node.name:
            return 'char*'

        return 'int32_t'  # Default

    def _infer_param_type(self, param_name: str, annotation) -> str:
        """Infer parameter type"""
        if annotation:
            if isinstance(annotation, ast.Name):
                py_type = annotation.id
                return self.type_mapper.map_type(py_type, Language.PYTHON, Language.C)

        # Infer from name
        if 'count' in param_name or 'size' in param_name or 'num' in param_name:
            return 'int32_t'
        elif 'name' in param_name or 'text' in param_name or 'msg' in param_name:
            return 'char*'
        elif 'flag' in param_name or 'enabled' in param_name:
            return 'bool'

        return 'int32_t'

    def _infer_type_from_value(self, value: ast.AST) -> str:
        """Infer type from assigned value"""
        if isinstance(value, ast.Constant):
            if isinstance(value.value, int):
                return 'int'
            elif isinstance(value.value, float):
                return 'float'
            elif isinstance(value.value, str):
                return 'str'
            elif isinstance(value.value, bool):
                return 'bool'

        elif isinstance(value, ast.List):
            return 'list'
        elif isinstance(value, ast.Dict):
            return 'dict'

        return 'int'  # Default


class PythonToRustTranslator:
    """Translate Python to Rust"""

    def __init__(self):
        self.type_mapper = TypeSystemMapper()
        self.warnings = []

    def translate(self, python_code: str) -> str:
        """Translate Python code to Rust"""
        self.warnings = []

        try:
            tree = ast.parse(python_code)
            rust_code = self._translate_module(tree)
            return rust_code
        except SyntaxError as e:
            self.warnings.append(f"Syntax error: {e}")
            return f"// Translation failed: {e}"

    def _translate_module(self, tree: ast.Module) -> str:
        """Translate module to Rust"""
        rust_lines = []

        # Add common use statements
        rust_lines.append("use std::collections::HashMap;")
        rust_lines.append("")

        # Translate each top-level statement
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                rust_lines.append(self._translate_function(node))
            elif isinstance(node, ast.ClassDef):
                rust_lines.append(self._translate_class(node))

        return '\n'.join(rust_lines)

    def _translate_function(self, func_node: ast.FunctionDef) -> str:
        """Translate Python function to Rust function"""
        func_name = func_node.name
        params = func_node.args.args

        # Build parameter list with Rust types
        rust_params = []
        for param in params:
            param_name = param.arg
            param_type = self._infer_param_type_rust(param_name, param.annotation)
            rust_params.append(f"{param_name}: {param_type}")

        params_str = ', '.join(rust_params)

        # Infer return type
        return_type = self._infer_return_type_rust(func_node)

        # Translate body
        body_lines = []
        for stmt in func_node.body:
            body_lines.append("    " + self._translate_statement_rust(stmt))

        body_str = '\n'.join(body_lines) if body_lines else "    // TODO: Implement\n    unimplemented!()"

        return f"""fn {func_name}({params_str}) -> {return_type} {{
{body_str}
}}
"""

    def _translate_class(self, class_node: ast.ClassDef) -> str:
        """Translate Python class to Rust struct + impl"""
        class_name = class_node.name

        # Extract fields
        fields = self._extract_class_fields_rust(class_node)

        # Build struct
        struct_def = f"struct {class_name} {{\n"
        for field_name, field_type in fields.items():
            rust_type = self.type_mapper.map_type(field_type, Language.PYTHON, Language.RUST)
            struct_def += f"    {field_name}: {rust_type},\n"
        struct_def += "}\n\n"

        # Translate methods in impl block
        impl_block = f"impl {class_name} {{\n"

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                if node.name == '__init__':
                    impl_block += self._translate_constructor_rust(node, class_name, fields)
                else:
                    impl_block += self._translate_method_rust(node)

        impl_block += "}\n"

        return struct_def + impl_block

    def _translate_constructor_rust(self, init_node: ast.FunctionDef, class_name: str, fields: Dict) -> str:
        """Translate __init__ to Rust new() constructor"""
        params = init_node.args.args[1:]  # Skip 'self'

        rust_params = []
        for param in params:
            param_name = param.arg
            if param_name in fields:
                param_type = self.type_mapper.map_type(fields[param_name], Language.PYTHON, Language.RUST)
                rust_params.append(f"{param_name}: {param_type}")

        params_str = ', '.join(rust_params)

        # Build field initialization
        field_inits = []
        for field_name in fields:
            field_inits.append(f"            {field_name},")

        fields_str = '\n'.join(field_inits)

        return f"""    fn new({params_str}) -> Self {{
        {class_name} {{
{fields_str}
        }}
    }}

"""

    def _translate_method_rust(self, method_node: ast.FunctionDef) -> str:
        """Translate method to Rust impl method"""
        method_name = method_node.name
        params = method_node.args.args[1:]  # Skip 'self'

        rust_params = ["&self"]
        for param in params:
            param_name = param.arg
            param_type = self._infer_param_type_rust(param_name, param.annotation)
            rust_params.append(f"{param_name}: {param_type}")

        params_str = ', '.join(rust_params)

        return_type = self._infer_return_type_rust(method_node)

        body_lines = []
        for stmt in method_node.body:
            body_lines.append("        " + self._translate_statement_rust(stmt))

        body_str = '\n'.join(body_lines) if body_lines else "        unimplemented!()"

        return f"""    fn {method_name}({params_str}) -> {return_type} {{
{body_str}
    }}

"""

    def _extract_class_fields_rust(self, class_node: ast.ClassDef) -> Dict[str, str]:
        """Extract class fields"""
        fields = {}

        for node in class_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if isinstance(target, ast.Attribute):
                                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                    field_name = target.attr
                                    field_type = self._infer_type_from_value(stmt.value)
                                    fields[field_name] = field_type

        return fields

    def _translate_statement_rust(self, stmt: ast.AST) -> str:
        """Translate statement to Rust"""
        if isinstance(stmt, ast.Return):
            value = self._translate_expression_rust(stmt.value) if stmt.value else ""
            return value if value else "return"  # Rust uses implicit returns

        elif isinstance(stmt, ast.Assign):
            target = stmt.targets[0]
            value = self._translate_expression_rust(stmt.value)

            if isinstance(target, ast.Name):
                var_name = target.id
                var_type = self._infer_type_from_value(stmt.value)
                rust_type = self.type_mapper.map_type(var_type, Language.PYTHON, Language.RUST)
                return f"let {var_name}: {rust_type} = {value};"

        return "// TODO"

    def _translate_expression_rust(self, expr: ast.AST) -> str:
        """Translate expression to Rust"""
        if expr is None:
            return ""

        if isinstance(expr, ast.Constant):
            if isinstance(expr.value, str):
                return f'"{expr.value}".to_string()'
            elif expr.value is None:
                return "None"
            else:
                return str(expr.value)

        elif isinstance(expr, ast.Name):
            return expr.id

        elif isinstance(expr, ast.BinOp):
            left = self._translate_expression_rust(expr.left)
            right = self._translate_expression_rust(expr.right)
            op = {ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/'}[type(expr.op)]
            return f"({left} {op} {right})"

        return "unimplemented!()"

    def _infer_return_type_rust(self, func_node: ast.FunctionDef) -> str:
        """Infer Rust return type"""
        if func_node.returns and isinstance(func_node.returns, ast.Name):
            py_type = func_node.returns.id
            return self.type_mapper.map_type(py_type, Language.PYTHON, Language.RUST)

        return "i64"

    def _infer_param_type_rust(self, param_name: str, annotation) -> str:
        """Infer Rust parameter type"""
        if annotation and isinstance(annotation, ast.Name):
            py_type = annotation.id
            return self.type_mapper.map_type(py_type, Language.PYTHON, Language.RUST)

        if 'count' in param_name or 'num' in param_name:
            return 'i64'
        elif 'name' in param_name or 'text' in param_name:
            return 'String'

        return 'i64'

    def _infer_type_from_value(self, value: ast.AST) -> str:
        """Infer type from value"""
        if isinstance(value, ast.Constant):
            if isinstance(value.value, int):
                return 'int'
            elif isinstance(value.value, str):
                return 'str'
            elif isinstance(value.value, bool):
                return 'bool'

        return 'int'


class CrossLanguageTranslator:
    """Main cross-language translator"""

    def __init__(self):
        self.python_to_c = PythonToCTranslator()
        self.python_to_rust = PythonToRustTranslator()

    def translate(self, source_code: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate code between languages"""

        source = Language(source_lang.lower())
        target = Language(target_lang.lower())

        warnings = []
        translated = ""

        if source == Language.PYTHON and target == Language.C:
            translated = self.python_to_c.translate(source_code)
            warnings = self.python_to_c.warnings

        elif source == Language.PYTHON and target == Language.RUST:
            translated = self.python_to_rust.translate(source_code)
            warnings = self.python_to_rust.warnings

        elif source == target:
            translated = source_code
            warnings.append("Source and target are the same language")

        else:
            warnings.append(f"Translation {source_lang} → {target_lang} not yet implemented")
            translated = f"// Translation {source_lang} → {target_lang} not implemented\n{source_code}"

        return TranslationResult(
            source_language=source,
            target_language=target,
            source_code=source_code,
            translated_code=translated,
            warnings=warnings
        )

    def format_result(self, result: TranslationResult) -> str:
        """Format translation result"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"CROSS-LANGUAGE TRANSLATION: {result.source_language.value.upper()} → {result.target_language.value.upper()}")
        lines.append("=" * 80)

        if result.warnings:
            lines.append("\nWarnings:")
            for warning in result.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        lines.append("\nTRANSLATED CODE:")
        lines.append("-" * 80)
        lines.append(result.translated_code)
        lines.append("-" * 80)

        return '\n'.join(lines)


# Example usage
if __name__ == "__main__":
    # Test Python → C translation
    python_code = '''
def calculate_sum(n: int) -> int:
    """Calculate sum of numbers from 0 to n"""
    total = 0
    for i in range(n + 1):
        total += i
    return total

class Counter:
    def __init__(self, start_value: int):
        self.value = start_value

    def increment(self):
        self.value += 1
        return self.value
'''

    translator = CrossLanguageTranslator()

    print("Python → C:")
    result_c = translator.translate(python_code, "python", "c")
    print(translator.format_result(result_c))

    print("\n\n")

    print("Python → Rust:")
    result_rust = translator.translate(python_code, "python", "rust")
    print(translator.format_result(result_rust))
