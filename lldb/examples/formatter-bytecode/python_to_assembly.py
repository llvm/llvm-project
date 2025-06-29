#!/usr/bin/python3

import ast
import io
import sys
from typing import Any

BUILTINS = {
    "Cast": "@cast",
    "GetChildMemberWithName": "@get_child_with_name",
    "GetSummary": "@get_summary",
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

class Compiler(ast.NodeVisitor):
    # Track the stack index of locals variables.
    #
    # This is essentially an ordered dictionary, where the key is an index on
    # the stack, and the value is the name of the variable whose value is at
    # that index.
    #
    # Ex: `locals[0]` is the name of the first value pushed on the stack, etc.
    locals: list[str]

    buffer: io.StringIO

    def __init__(self) -> None:
        self.locals = []
        self.buffer = io.StringIO()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Initialize `locals` with the (positional) arguments.
        self.locals = [arg.arg for arg in node.args.args]
        self.generic_visit(node)
        self.locals.clear()

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        # XXX: Does not handle multiple comparisons, ex: `0 < x < 10`
        self.visit(node.comparators[0])
        self._output(COMPS[type(node.ops[0])])

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)

        self._output("{")
        self._visit_each(node.body)
        if node.orelse:
            self._output("} {")
            self._visit_each(node.orelse)
            self._output("} ifelse")
        else:
            self._output("} if")

    def visit_Return(self, node: ast.Return) -> None:
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

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            # The receiver is the left hande side of the dot.
            receiver = node.func.value
            method = node.func.attr
            if selector := BUILTINS.get(method):
                # Visit the method's receiver to have its value on the stack.
                self.visit(receiver)
                # Visit the args to position them on the stack.
                self._visit_each(node.args)
                self._output(f"{selector} call")
            else:
                # TODO: fail
                print(f"error: unsupported method {node.func.attr}", file=sys.stderr)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Visit RHS first, putting values on the stack.
        self.visit(node.value)
        # Determine the name(s). Either a single Name, or a Tuple of Names.
        target = node.targets[0]
        if isinstance(target, ast.Name):
            names = [target.id]
        elif isinstance(target, ast.Tuple):
            # These tuple elements are Name nodes.
            names = [x.id for x in target.elts]

        # Forget any previous bindings of these names.
        # Their values are orphaned on the stack.
        for local in self.locals:
            if local in names:
                old_idx = self.locals.index(local)
                self.locals[old_idx] = ""

        self.locals.extend(names)

    def visit_Name(self, node: ast.Name) -> None:
        idx = self.locals.index(node.id)
        self._output(f"{idx} pick # {node.id}")

    def _visit_each(self, nodes: list[ast.AST]) -> None:
        for child in nodes:
            self.visit(child)

    def _output(self, x: Any) -> None:
        print(x, file=self.buffer)

    @property
    def output(self) -> str:
        return compiler.buffer.getvalue()


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        root = ast.parse(f.read())
    compiler = Compiler()
    compiler.visit(root)
    print(compiler.output)
