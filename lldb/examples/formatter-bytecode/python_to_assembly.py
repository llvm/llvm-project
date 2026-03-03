#!/usr/bin/python3

import ast
import io
import sys
from copy import copy
from typing import Any, Optional, Sequence, Union, cast

BUILTINS = {
    "Cast": "@cast",
    "GetChildAtIndex": "@get_child_at_index",
    "GetChildMemberWithName": "@get_child_with_name",
    "GetSummary": "@summary",
    "GetSyntheticValue": "@get_synthetic_value",
    "GetTemplateArgumentType": "@get_template_argument_type",
    "GetType": "@get_type",
    "GetValueAsUnsigned": "@get_value_as_unsigned",
}

COMPS = {
    ast.Eq: "=",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "=<",
    ast.Gt: ">",
    ast.GtE: "=>",
}

# Maps Python method names in a formatter class to their bytecode signatures.
METHOD_SIGS = {
    "__init__": "@init",
    "update": "@update",
    "num_children": "@get_num_children",
    "get_child_index": "@get_child_index",
    "get_child_at_index": "@get_child_at_index",
    "get_value": "@get_value",
}


class CompilerError(Exception):
    lineno: int

    def __init__(self, message, node: Union[ast.expr, ast.stmt]) -> None:
        super().__init__(message)
        self.lineno = node.lineno


class Compiler(ast.NodeVisitor):
    # Names of locals in bottom-to-top stack order. locals[0] is the
    # oldest/deepest; locals[-1] is the most recently pushed.
    locals: list[str]

    # Names of visible attrs in bottom-to-top stack order. Always holds the
    # full combined frame for the method being compiled: grows incrementally
    # during __init__/update, and is set to the combined list before getter
    # methods are compiled.
    attrs: list[str]

    # Temporaries currently on the stack above the locals/attrs frame.
    # Always 0 at statement boundaries.
    num_temps: int

    # Bytecode signature of the method being compiled, or None for top-level
    # functions.
    current_sig: Optional[str]

    buffer: io.StringIO

    def __init__(self) -> None:
        self.locals = []
        self.attrs = []
        self.num_temps = 0
        self.current_sig = None
        self.buffer = io.StringIO()

    def compile(self, source_file: str) -> str:
        with open(source_file) as f:
            root = ast.parse(f.read())
        self.visit(root)
        return self.buffer.getvalue()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Compile methods in a fixed order so that attrs is fully populated
        # before getter methods are compiled.
        methods = {}
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name not in METHOD_SIGS:
                    raise CompilerError(f"unsupported method: {item.name}", item)
                methods[item.name] = item

        self.attrs = []
        if method := methods.get("__init__"):
            self._compile_method(method)
        # self.attrs now holds init's attrs. update's attrs are appended above
        # them, so after update self.attrs is the combined init+update list.
        if method := methods.get("update"):
            self._compile_method(method)

        for method_name, method in methods.items():
            if method_name not in ("__init__", "update"):
                self._compile_method(method)

    def _compile_method(self, node: ast.FunctionDef) -> None:
        self.current_sig = METHOD_SIGS[node.name]
        self.num_temps = 0

        # Strip 'self' (and 'internal_dict' for __init__) from the arg list;
        # the remaining args become the initial locals.
        args = copy(node.args.args)
        args.pop(0)  # drop 'self'
        if node.name == "__init__":
            args.pop()  # drop trailing 'internal_dict'

        self.locals = [arg.arg for arg in args]

        # Compile into a temporary buffer so the signature line can be
        # emitted first.
        saved_buffer = self.buffer
        self.buffer = io.StringIO()

        self._visit_each(node.body)

        method_output = self.buffer.getvalue()
        self.buffer = saved_buffer
        self._output(f"@{self.current_sig}:")
        self._output(method_output)

        self.locals.clear()
        self.current_sig = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Top-level function (not inside a class).
        self.current_sig = None
        self.attrs = []
        self.locals = [arg.arg for arg in node.args.args]
        self._visit_each(node.body)
        self.locals.clear()

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        # XXX: Does not handle multiple comparisons, ex: `0 < x < 10`
        self.visit(node.comparators[0])
        self._output(COMPS[type(node.ops[0])])
        # The comparison consumes two values and produces one.
        self.num_temps -= 1

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        # `if`/`ifelse` consumes the condition.
        self.num_temps = 0

        self._output("{")
        self._visit_each(node.body)
        if node.orelse:
            self.num_temps = 0
            self._output("} {")
            self._visit_each(node.orelse)
            self._output("} ifelse")
        else:
            self._output("} if")

    def visit_Return(self, node: ast.Return) -> None:
        self.num_temps = 0
        if node.value:
            self.visit(node.value)
        self._output("return")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str):
            self._output(f'"{node.value}"')
        elif isinstance(node.value, bool):
            self._output(int(node.value))
        else:
            self._output(node.value)
        self.num_temps += 1

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            receiver = func.value
            method = func.attr
            # self is not a valid call receiver.
            if isinstance(receiver, ast.Name) and receiver.id == "self":
                raise CompilerError(
                    "self is not a valid call receiver; use self.attr to read an attribute",
                    node,
                )
            if selector := BUILTINS.get(method):
                self.visit(receiver)
                self._visit_each(node.args)
                self._output(f"{selector} call")
                # `call` pops the receiver and all args, and pushes one result.
                self.num_temps -= len(node.args)
                return
            raise CompilerError(f"unsupported method: {method}", node)

        if isinstance(func, ast.Name):
            raise CompilerError(f"unsupported function: {func.id}", node)

        raise CompilerError("unsupported function call expression", node)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.num_temps = 0

        target = node.targets[0]

        # Handle self.attr = expr (attribute assignment).
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            if self.current_sig not in ("@init", "@update"):
                raise CompilerError(
                    "attribute assignment is only allowed in __init__ and update",
                    node,
                )

            attr = target.attr
            if attr in self.attrs:
                raise CompilerError(f"attribute '{attr}' is already assigned", node)

            # If the RHS is an argument (the only kind of local permitted in
            # __init__) - then it is already on the stack in place, and no
            # evaluation is needed.
            is_arg = (
                isinstance(node.value, ast.Name)
                and self._local_index(node.value) is not None
            )
            if not is_arg:
                # Evaluate the RHS, leaving its value on the stack.
                self.visit(node.value)

            # Record the attr.
            self.attrs.append(attr)
            return

        # Handle local variable assignment.
        if self.current_sig in ("@init", "@update"):
            raise CompilerError(
                "local variable assignment is not allowed in __init__ or update; "
                "use attribute assignment (self.attr = ...) instead",
                node,
            )

        if isinstance(target, ast.Name):
            names = [target]
        elif isinstance(target, ast.Tuple):
            names = cast(list[ast.Name], target.elts)
        else:
            raise CompilerError("unsupported assignment target", node)

        # Visit RHS, leaving its value on the stack.
        self.visit(node.value)

        # Forget any previous bindings of these names.
        # Their values are orphaned on the stack.
        for name in names:
            idx = self._local_index(name)
            if idx is not None:
                self.locals[idx] = ""

        self.locals.extend(x.id for x in names)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Only self.attr reads are supported here.
        if not (isinstance(node.value, ast.Name) and node.value.id == "self"):
            raise CompilerError(
                "unsupported attribute access (only self.attr is supported)", node
            )
        attr_idx = self._attr_index(node.attr, node)
        pick_idx = self.num_temps + attr_idx
        self._output(f"{pick_idx} pick # self.{node.attr}")
        self.num_temps += 1

    def visit_Name(self, node: ast.Name) -> None:
        idx = self._stack_index(node)
        if idx is None:
            raise CompilerError(f"unknown local variable: {node.id}", node)
        self._output(f"{idx} pick # {node.id}")
        self.num_temps += 1

    def _visit_each(self, nodes: Sequence[ast.AST]) -> None:
        for child in nodes:
            self.visit(child)

    def _attr_index(self, name: str, node: ast.expr) -> int:
        # self.attrs is always the full visible attr frame, so the index is
        # the direct pick offset with no further adjustment.
        try:
            return self.attrs.index(name)
        except ValueError:
            raise CompilerError(f"unknown attribute: {name}", node)

    def _stack_index(self, name: ast.Name) -> Optional[int]:
        # Offset past all attrs and any in-flight temporaries.
        idx = self._local_index(name)
        if idx is None:
            return None
        return len(self.attrs) + idx + self.num_temps

    def _local_index(self, name: ast.Name) -> Optional[int]:
        try:
            return self.locals.index(name.id)
        except ValueError:
            return None

    def _output(self, x: Any) -> None:
        print(x, file=self.buffer)


if __name__ == "__main__":
    source_file = sys.argv[1]
    compiler = Compiler()
    try:
        output = compiler.compile(source_file)
        print(output)
    except CompilerError as e:
        print(f"{source_file}:{e.lineno}: {e}", file=sys.stderr)
