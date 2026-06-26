# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for using debugger output to generate expected values that match that output."""

from collections import Counter, OrderedDict, defaultdict
from copy import deepcopy
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dex.dextIR import DextIR, StepIR, ValueIR
from dex.evaluation.StateMatch import get_active_where_matches
from dex.test_script.Nodes import Expect, Then, Value, Where
from dex.test_script.Script import DexterScript, Scope
from dex.tools.Main import Context


class ExpectedValueRewriter:
    """Given a ValueIR for an Expect, generates a complete expected value that matches that value if one can be
    provided."""

    def __init__(self, expect: Expect, value: ValueIR):
        self.expect = expect
        self.root_value = value
        self.expected_value = expect.get_variable_result(value)


def unique_expected_values(elements: List[ExpectedValueRewriter]):
    """Given a list of ExpectedValueRewriters, and returns either a list containing the unique set of non-None expected
    values, or a single item if there is only one non-duplicated expected value in the list, or None if there are no
    valid expected values."""

    unique_set = set()
    result = []
    for element in elements:
        expected_value = element.expected_value
        if expected_value is None:
            continue
        if expected_value not in unique_set:
            unique_set.add(expected_value)
            result.append(expected_value)
    if not result:
        return None
    if len(result) == 1:
        return result[0]
    return result


class StepExpectRewriter:
    """Processes all active, unknown expects at a given debugger step and produces ExpectedValueRewriter results for
    each."""

    def __init__(self, step: StepIR, script: DexterScript):
        self.step = step
        self.script = script
        self.state_match = get_active_where_matches(script, step)
        active_expects = {
            expect
            for where_match in self.state_match.values()
            for expect in where_match.active_expects
        }
        self.expect_matches: Dict[Expect, ExpectedValueRewriter] = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            assert isinstance(expect, Value), "Non-Value expects currently unsupported"
            if expect in active_expects and expected_value is None:
                self.expect_matches[expect] = ExpectedValueRewriter(
                    expect, step.watches[expect.get_watched_expr()]
                )

        script.visit_script(visit_expect=add_expected_values)


class ScriptExpectRewriter:
    """Given the full output from a debugger run and a script with missing expected values, returns a script with
    filled-in expected values that match the debugger output."""

    def __init__(self, context: Context, dext_ir: DextIR):
        self.context = context
        self.dext_ir = dext_ir
        self.unknown_expect_rewrites: Dict[
            Expect, List[Tuple[int, ExpectedValueRewriter]]
        ] = {}
        self.new_script: Optional[DexterScript] = None
        self.new_expected_values: Dict[Expect, Any] = {}
        self.missing_expect_rewrites: List[Expect] = []

        script = dext_ir.script
        assert (
            script is not None
        ), "Cannot use ScriptExpectRewriter on a non-script Dexter test."

        # Collect every Expect with an unknown value into the `unknown_expect_rewrites` dict. We expect all Expects in
        # this dict to have observed values, and don't expect to rewrite any Expects outside of this dict.
        def collect_unknown_expects(expect: Expect, expected_value: Any, scope: Scope):
            assert isinstance(expect, Value), "Non-Value expects currently unsupported"
            if expected_value is None:
                self.unknown_expect_rewrites[expect] = []

        script.visit_script(visit_expect=collect_unknown_expects)

        # If there are no expects to update, then there is no rewriting to be done - exit early.
        if not self.unknown_expect_rewrites:
            return

        # Populate the `unknown_expect_rewrites` dict, mapping each expect with an unknown value to its list of observed
        # during this run, along with the corresponding step indices.
        self.step_rewriters = [
            StepExpectRewriter(step, script) for step in dext_ir.steps
        ]
        for step_rewriter in self.step_rewriters:
            step_idx = step_rewriter.step.step_index
            for expect, expected_value_rewriter in step_rewriter.expect_matches.items():
                self.unknown_expect_rewrites[expect].append(
                    (step_idx, expected_value_rewriter)
                )

        # For each unknown expect, merge the observed values into a writable "expected values" entry, which may be a
        # list or a single value.
        self.new_expected_values = {
            expect: expected_values
            for expect, expect_rewriters in self.unknown_expect_rewrites.items()
            if (
                expected_values := unique_expected_values(
                    [rewriter for idx, rewriter in expect_rewriters]
                )
            )
            is not None
        }

        # Finally, use the new expected values to rewrite the script.
        self.new_script = rewrite_script(script, self.new_expected_values)
        self.missing_expect_rewrites = [
            expect
            for expect in self.unknown_expect_rewrites
            if expect not in self.new_expected_values
        ]

    @property
    def num_successful_rewrites(self):
        return len(self.new_expected_values)

    @property
    def num_unsuccessful_rewrites(self):
        return len(self.missing_expect_rewrites)


def rewrite_script(
    script: DexterScript, add_expected_values: Dict[Expect, Any]
) -> DexterScript:
    """Given a set of updates to apply to a provided script, returns a copy of the script_obj with the updates
    applied.
    Does not deep copy, meaning the new script contains the same node objects as the old script; this is safe as we do
    not modify these objects."""
    # First build up a map describing the children of every node in the script, adding add_expected_values to the
    # required expect nodes.
    new_node_child_map = {}

    def replace_where(where: Where, scope: Scope):
        if scope.where:
            scope_where_children = new_node_child_map.setdefault(scope.where, [])
            assert isinstance(
                scope_where_children, list
            ), f"Unexpected child for !where node: {scope_where_children}"
            scope_where_children.append(where)

    def replace_then(then: Then, scope: Scope):
        assert (
            scope.where not in new_node_child_map
        ), "!then must be the sole child of a state node."
        new_node_child_map[scope.where] = then

    def replace_expect(expect: Expect, expected_value, scope: Scope):
        new_expected_value = add_expected_values.get(expect) or expected_value
        new_node_child_map[expect] = new_expected_value
        scope_where_children = new_node_child_map.setdefault(scope.where, [])
        assert isinstance(
            scope_where_children, list
        ), f"Unexpected child for state node {scope.where}: {scope_where_children}"
        scope_where_children.append(expect)

    script.visit_script(
        visit_where=replace_where, visit_expect=replace_expect, visit_then=replace_then
    )

    # Now rebuild the script object using the two maps.
    def build_subscript(node):
        """Returns the subset of the script object whose parent is the given node."""
        assert isinstance(
            node, (Expect, Where)
        ), f"Unexpected script parent node: {node}"
        if isinstance(node, Expect):
            return new_node_child_map[node]
        node_children = new_node_child_map[node]
        if isinstance(node_children, Then):
            return node_children
        assert isinstance(
            node_children, List
        ), f"Unexpected child for state node {node}: {node_children}"
        return {child: build_subscript(child) for child in node_children}

    new_script_obj = {node: build_subscript(node) for node in script.script_obj}
    return DexterScript(
        script.context,
        new_script_obj,
        script.root_scope,
        script.base_dir,
        script.load_context,
    )
