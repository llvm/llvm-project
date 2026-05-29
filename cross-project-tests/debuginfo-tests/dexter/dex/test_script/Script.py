# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""This file defines the DexterScript class. Using Nodes as building blocks, the DexterScript defines a complete Dexter
test, a structured definition of locations, values, and actions used to drive a debugging session and evaluate the
results.
"""

from pathlib import PurePath
import os
from typing import Any, Callable, Optional, Set
import yaml

from dex.test_script.Nodes import (
    Expect,
    Where,
    setup_yaml_parser,
)

from dex.utils.Exceptions import Error
from dex.utils.Timer import Timer


class DexterScriptError(Error):
    pass


class Scope:
    """Helper class used to simplify queries about the context of a Node in the Dexter Script. The context for a given
    Node consists of some base context information in the root of the script, and then all Where nodes in the parent
    chain of the current Node. Therefore each Script has a root Scope object, and each Node's context is given by a
    Scope chain built from the root Scope and every Where between the root and the given Node.
    """

    def __init__(
        self,
        file: Optional[str] = None,
        where: Optional[Where] = None,
        parent_scope: "Optional[Scope]" = None,
    ):
        """Can be initialized with either a file for the default Scope, or with the properties of a Where
        for any script-nested Scope.
        """
        if where is not None:
            assert (
                parent_scope is not None
            ), "Scope for a Where node must have a parent scope!"
            assert (
                file is None
            ), "Scope for a Where node cannot have a separately-defined file!"
            self.file = None
            self.where = where
            self.parent_scope = parent_scope
        else:
            assert (
                parent_scope is None
            ), "Scope for a Root node cannot have a parent scope!"
            self.file = file
            self.where = None
            self.parent_scope = None

    def add_where(self, where: Where):
        """Adds `where` to this Scope's chain."""
        return Scope(where=where, parent_scope=self)


class DexterScript:
    def __init__(
        self,
        context,
        script_obj,
        scope: Scope,
    ):
        self.context = context
        self.script_obj = script_obj
        self.root_scope = scope
        # `visit_script` will validate the structure of the script, as it traverses the full script and raises an
        # exception if it sees anything unexpected.
        self.visit_script()

    # If a truthy value is returned, abort further visiting and return that value.
    def _visit_script(
        self, script, scope: Scope, visit_where=None, visit_expect=None, visit_then=None
    ) -> Any:
        def do(visitor, *args):
            if visitor:
                return visitor(*args)
            return None

        if not isinstance(script, dict):
            raise DexterScriptError(f"Found unexpected node: {script}")
        for key, value in script.items():
            if isinstance(key, Where):
                if result := do(visit_where, key, scope):
                    return result
                new_scope = scope.add_where(key)
                if result := self._visit_script(
                    value, new_scope, visit_where, visit_expect, visit_then
                ):
                    return result
            elif isinstance(key, Expect):
                if result := do(visit_expect, key, value, scope):
                    return result
            else:
                raise DexterScriptError(f"Found unexpected node: {key}")

    # Any visitor function provided may return a truthy value to abort the visit and return that value.
    def visit_script(
        self,
        visit_where: Optional[Callable[[Where, Scope], Any]] = None,
        visit_expect: Optional[Callable[[Expect, Any, Scope], Any]] = None,
    ) -> Any:
        """Visits all nodes in the script in pre-order traversal, calling any non-none provided visitor functions for
        each respective node type. Note that we do not visit expected values independently of their associated expect;
        instead, visit_expect accepts the Expect node and its expected value as an argument.

        If any visit function returns a truthy value, traversal will early-exit and this function returns that value;
        otherwise, this function returns None."""
        return self._visit_script(
            self.script_obj, self.root_scope, visit_where, visit_expect
        )

    @property
    def root_wheres(self) -> Set[Where]:
        return set(node for node in self.script_obj if isinstance(node, Where))

    def dump(self) -> str:
        return yaml.dump(self.script_obj)


# Helper function to apply a line offset to the errors reported by YAML while loading, to account for the YAML documents
# being embedded in part of a file.
def try_load_yaml(yaml_doc, loader, line_offset=0):
    """Helper function that loads a YAML document from within a file, where the document may start in the middle of the
    file. In this case, the value of line_offset should be set to the start line of the YAML document, and this function
    will fix-up any returned syntax errors to point to the correct line in the file."""
    try:
        return yaml.load(yaml_doc, loader)
    except yaml.MarkedYAMLError as e:
        # MarkedYAMLError is an error with a 'Mark' pointing to the location of the error; this helper function applies
        # our line offset to the provided mark if it is present.
        def adjust_mark_loc(mark: Optional[yaml.Mark]) -> Optional[yaml.Mark]:
            if mark is None:
                return None
            return yaml.Mark(
                mark.name,
                mark.index,
                mark.line + line_offset,
                mark.column,
                mark.buffer,
                mark.pointer,
            )

        # Adjust the error marks and then propagate the adjusted error.
        e.context_mark = adjust_mark_loc(e.context_mark)
        e.problem_mark = adjust_mark_loc(e.problem_mark)
        raise e


def get_script(context, file, loader) -> DexterScript:
    """Searches the given file for a valid Dexter script, and returns the first valid script that it finds or raises an
    Error if none is found."""
    if not os.path.exists(file):
        raise Error(f"Provided script file '{file}' does not exist.")
    with open(file, "r") as r:
        lines = r.readlines()
    if not lines:
        raise Error(f"Provided script file '{file}' is empty.")

    numbered_lines = [(idx + 1, line) for idx, line in enumerate(lines)]
    root_scope = Scope(file=str(file))
    start_line = None
    attempted_scripts = []
    start_line = next((idx for idx, line in numbered_lines if line == "---\n"), None)
    if start_line is None:
        # If we saw no '---', then assume the whole file is a document and try to parse it.
        try:
            return DexterScript(
                context,
                try_load_yaml("\n".join(lines), loader),
                root_scope,
            )
        except (Error, yaml.YAMLError) as e:
            raise Error(f"File '{file}' was not a valid Dexter script:\n{e}")
    # If we have at least one valid document start, then check every document until we see one that is a valid Dexter
    # test.
    while start_line is not None:
        stop_line = next(
            (
                idx
                for idx, line in numbered_lines[start_line + 1 :]
                if line.startswith("...")
            ),
            len(lines),
        )
        try:
            return DexterScript(
                context,
                try_load_yaml(
                    "\n".join(lines[start_line:stop_line]), loader, start_line
                ),
                root_scope,
            )
        except (Error, yaml.YAMLError) as e:
            attempted_scripts.append((start_line, e))
        start_line = next(
            (idx for idx, line in numbered_lines[stop_line + 1 :] if line == "---\n"),
            None,
        )
    script_error_messages = "\n".join(
        f"Script starting line {line}:\n{e}" for line, e in attempted_scripts
    )
    raise Error(
        f"No valid Dexter script found in file '{file}'; candidates:\n{script_error_messages}"
    )


def get_dexter_script(context, test_file, source_root_dir):
    setup_yaml_parser(yaml.CLoader)
    with Timer("parsing script"):
        script = get_script(context, test_file, yaml.CLoader)
        assert script.root_scope.file == test_file
        source_files = set()
        source_dir = source_root_dir if source_root_dir else str(test_file)

        def check_explicit_files(where: Where, _: Scope):
            if not where.file:
                return
            declared_path = where.file
            if not os.path.isabs(declared_path):
                declared_path = os.path.join(source_dir, declared_path)
            source_files.add(str(PurePath(declared_path)))

        script.visit_script(visit_where=check_explicit_files)
        return script, source_files
