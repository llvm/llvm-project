# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for using debugger output to generate expected values that match that output."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dex.dextIR import DextIR, StepIR, ValueIR
from dex.evaluation.StateMatch import StateMatchContext, get_state_match
from dex.test_script.Nodes import (
    DexRange,
    Expect,
    ExpectAll,
    Line,
    Step,
    Then,
    Type,
    Value,
    Where,
)
from dex.test_script.Script import DexterScript, Scope
from dex.tools.Main import Context
from dex.utils.Exceptions import Error


class ExpectedValueRewriter:
    """Given a ValueIR for an Expect, generates a complete expected value that matches that value if one can be
    provided."""

    def __init__(self, expect: Expect, value: ValueIR):
        self.expect = expect
        self.root_value = value
        self.expected_value: Union[Dict, str, None] = None
        if sub_values := self.root_value.sub_values:
            self.expected_value = {
                sub_value.expression: expected_value
                for sub_value in sub_values
                if (
                    expected_value := ExpectedValueRewriter(
                        expect, sub_value
                    ).expected_value
                )
                is not None
            }
        if not self.expected_value:
            self.expected_value = expect.get_variable_result(value)


def unique_expected_values(elements: List[ExpectedValueRewriter]):
    """Given a list of ExpectedValueRewriters, and returns either a list containing the unique set of non-None expected
    values, or a single item if there is only one non-duplicated expected value in the list, or None if there are no
    valid expected values."""

    def freeze(input):
        assert input is not None, "Unexpected 'None' in an expected_value"
        if isinstance(input, dict):
            return tuple(sorted((str(k), freeze(v)) for k, v in input.items()))
        return input

    unique_set = set()
    result = []
    for element in elements:
        expected_value = element.expected_value
        if expected_value is None:
            continue
        frozen_value = freeze(expected_value)
        if frozen_value not in unique_set:
            unique_set.add(frozen_value)
            result.append(expected_value)
    if not result:
        return None
    if len(result) == 1:
        return result[0]
    return result


class ExpectedScopeRewriter:
    """Given a list of ValueIRs for all variables in a scope, generates a set of expected values for each."""

    def __init__(self, expect: Expect, step: StepIR, values: List[ValueIR]):
        self.expect = expect
        self.step = step
        self.values = values
        self.expected_values = [
            ExpectedValueRewriter(expect, value) for value in values
        ]


# (StartLine, StopLine) -> [(Var, ExpectedValues)]
ExpectedScopeRewrites = Dict[Optional[Tuple[int, int]], List[Tuple[str, Any]]]


def collect_scope_values(
    step_scope_values: List[ExpectedScopeRewriter],
) -> ExpectedScopeRewrites:
    if not step_scope_values:
        return {}
    assert all(
        step_scope_values[0].step.current_location.path == sv.step.current_location.path
        for sv in step_scope_values[1:]
    ), "Dexter currently does not handle scope watches that span multiple files."
    all_vars: Set[str] = set()
    for step_scope_rewriter in step_scope_values:
        all_vars.update(
            ev_rewriter.root_value.expression
            for ev_rewriter in step_scope_rewriter.expected_values
        )
    line_sorted_steps = sorted(
        step_scope_values,
        key=lambda step_scope_rewriter: step_scope_rewriter.step.current_location.lineno,
    )

    # Now we have a list of expected values for each variable sorted by the lines at which they appear; we use this to
    # form blocks of continuous liveness. In general, for unoptimized code we expect variables to have a single
    # continuous live range.
    per_range_var_unique_expected_values: ExpectedScopeRewrites = defaultdict(list)
    for var in sorted(all_vars):
        # Now build a list of all continuous live ranges.
        continuous_live_ranges: List[Tuple[int, int, List[ExpectedValueRewriter]]] = []
        is_live = False
        ever_dead = False
        for step_scope_rewriter in line_sorted_steps:
            line = step_scope_rewriter.step.current_location.lineno
            var_ev_rewriter = next(
                (
                    ev_rewriter
                    for ev_rewriter in step_scope_rewriter.expected_values
                    if ev_rewriter.root_value.expression == var
                ),
                None,
            )
            if var_ev_rewriter is None or var_ev_rewriter.expected_value is None:
                is_live = False
                ever_dead = True
                continue

            if not is_live:
                continuous_live_ranges.append((line, line, [var_ev_rewriter]))
            else:
                start, stop, range_evs = continuous_live_ranges[-1]
                assert line >= stop
                range_evs.append(var_ev_rewriter)
                continuous_live_ranges[-1] = (start, line, range_evs)
            is_live = True

        if not continuous_live_ranges:
            continue

        if not ever_dead:
            assert len(continuous_live_ranges) == 1
            per_range_var_unique_expected_values[None].append(
                (var, unique_expected_values(continuous_live_ranges[0][2]))
            )
            continue

        # Finally, collect the results into the per_var map.
        for start, stop, expected_values in continuous_live_ranges:
            per_range_var_unique_expected_values[(start, stop)].append(
                (var, unique_expected_values(expected_values))
            )

    return per_range_var_unique_expected_values


def get_expected_lines(expect: Expect, lines: List[int]) -> List[int]:
    """For a !step expect and the list of lines seen while that expect was active, returns the list of lines that should
    be expected by that expect."""
    assert isinstance(expect, Step), "Trying to get expected lines for non-step node?"
    if expect.kind == "never":
        # We can't really get useful "expected values" for a !step never node, unless we throw in some convoluted extra
        # steps, e.g. finding all breakpoint locations within the expect's enclosing scope, and creating a list of all
        # lines that have valid breakpoint locations but weren't seen.
        return []
    # Although !step order and !step exactly are evaluated differently, they both aim to match lines stepped on; since
    # we don't have any meaningful reason to exclude any seen lines from the written expected line list, we just use the
    # whole thing.
    return lines


class StepExpectRewriter:
    """Processes all active, unknown expects at a given debugger step and produces ExpectedValueRewriter results for
    each."""

    def __init__(
        self, step: StepIR, script: DexterScript, state_match_context: StateMatchContext
    ):
        self.step = step
        self.script = script
        self.state_match = get_state_match(script, step, state_match_context)
        active_expects = {
            expect: where_match.frame_idx
            for where_match in self.state_match.where_match_results.values()
            for expect in where_match.active_expects
        }
        self.expect_value_matches: Dict[Expect, ExpectedValueRewriter] = {}
        self.expect_scope_matches: Dict[Expect, ExpectedScopeRewriter] = {}
        self.expect_step_matches: Dict[Expect, int] = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            if expect not in active_expects or expected_value is not None:
                return
            expect_frame_idx = active_expects[expect]
            if (expr := expect.get_watched_expr()) is not None:
                self.expect_value_matches[expect] = ExpectedValueRewriter(
                    expect, step.frames[expect_frame_idx].watches[expr]
                )
            elif (scope_name := expect.get_watched_scope()) is not None:
                scope_vars = step.frames[expect_frame_idx].scope_watches.get(
                    scope_name, []
                )
                self.expect_scope_matches[expect] = ExpectedScopeRewriter(
                    expect,
                    step,
                    [step.frames[expect_frame_idx].watches[var] for var in scope_vars],
                )
            elif isinstance(expect, Step):
                self.expect_step_matches[expect] = step.current_location.lineno
            else:
                raise Error(
                    f"Unexpected expect without watched expression or scope: {expect}"
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
        self.scope_expect_rewrites: Dict[
            Expect, List[Tuple[int, ExpectedScopeRewriter]]
        ] = {}
        self.step_expect_rewrites: Dict[Expect, List[Tuple[int, int]]] = {}
        self.new_script: Optional[DexterScript] = None
        self.new_expected_values: Dict[Expect, Any] = {}
        self.new_expected_scopes: Dict[Expect, ExpectedScopeRewrites] = {}
        self.missing_expect_rewrites: List[Expect] = []

        script = dext_ir.script
        assert (
            script is not None
        ), "Cannot use ScriptExpectRewriter on a non-script Dexter test."

        # Collect every Expect with an unknown value into the `unknown_expect_rewrites` dict. We expect all Expects in
        # this dict to have observed values, and don't expect to rewrite any Expects outside of this dict.
        def collect_expects_to_rewrite(
            expect: Expect, expected_value: Any, scope: Scope
        ):
            if expected_value is not None:
                return
            if isinstance(expect, ExpectAll):
                self.scope_expect_rewrites[expect] = []
                return
            if isinstance(expect, Step):
                self.step_expect_rewrites[expect] = []
                return
            assert (
                expect.get_watched_expr() is not None
            ), f"Unexpected expect node kind {expect}"
            self.unknown_expect_rewrites[expect] = []

        script.visit_script(visit_expect=collect_expects_to_rewrite)

        # If there are no expects to update, then there is no rewriting to be done - exit early.
        if (
            not self.unknown_expect_rewrites
            and not self.scope_expect_rewrites
            and not self.step_expect_rewrites
        ):
            return

        def check_condition(step: StepIR, frame_idx: int, condition: str):
            cond_value = step.frames[frame_idx].watches[condition]
            result = cond_value.could_evaluate and cond_value.value.lower() == "true"
            return result

        state_match_context = StateMatchContext(check_condition=check_condition)
        self.step_rewriters = [
            StepExpectRewriter(step, script, state_match_context)
            for step in dext_ir.steps
        ]
        # Populate the expect_rewrites dicts, mapping each expect with an unknown value to its list of observed values
        # during this run, along with the corresponding step indices.
        for step_rewriter in self.step_rewriters:
            step_idx = step_rewriter.step.step_index
            for (
                expect,
                expected_value_rewriter,
            ) in step_rewriter.expect_value_matches.items():
                self.unknown_expect_rewrites[expect].append(
                    (step_idx, expected_value_rewriter)
                )
            for (
                expect,
                expected_scope_rewriter,
            ) in step_rewriter.expect_scope_matches.items():
                self.scope_expect_rewrites[expect].append(
                    (step_idx, expected_scope_rewriter)
                )
            for (
                expect,
                line,
            ) in step_rewriter.expect_step_matches.items():
                self.step_expect_rewrites[expect].append((step_idx, line))

        # For each unknown expect, merge the observed values into a writable "expected values" entry, which may be a
        # list or a single value.
        self.new_expected_values: Dict[Expect, Any] = {
            expect: expected_values
            for expect, expect_rewriters in self.unknown_expect_rewrites.items()
            if (
                expected_values := unique_expected_values(
                    [rewriter for idx, rewriter in expect_rewriters]
                )
            )
            is not None
        }
        # Do the same for unknown step expects.
        self.new_expected_values.update(
            {
                expect: get_expected_lines(
                    expect, [line for step_index, line in step_lines]
                )
                for expect, step_lines in self.step_expect_rewrites.items()
            }
        )
        # Do the same for unknown scope expects.
        self.new_expected_scopes = {
            expect: collect_scope_values(
                [rewriter for idx, rewriter in expect_rewriters]
            )
            for expect, expect_rewriters in self.scope_expect_rewrites.items()
        }

        # Finally, use the new expected values to rewrite the script.
        self.new_script = rewrite_script(
            script, self.new_expected_values, self.new_expected_scopes
        )
        self.missing_expect_rewrites = [
            expect
            for expect in self.unknown_expect_rewrites
            if expect not in self.new_expected_values
        ]

    @property
    def num_successful_rewrites(self):
        return len(self.new_expected_values) + sum(
            sum(len(var_expects) for var_expects in new_expected_scope.values())
            for new_expected_scope in self.new_expected_scopes.values()
        )

    @property
    def num_unsuccessful_rewrites(self):
        return len(self.missing_expect_rewrites)


def rewrite_script(
    script: DexterScript,
    add_expected_values: Dict[Expect, Any],
    expected_scope_rewrites: Dict[Expect, ExpectedScopeRewrites],
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
        scope_where_children = new_node_child_map.setdefault(scope.where, [])
        assert isinstance(
            scope_where_children, list
        ), f"Unexpected child for state node {scope.where}: {scope_where_children}"
        if isinstance(expect, ExpectAll):
            assert (
                expect in expected_scope_rewrites
            ), "Script-rewriter error: Dexter missed rewriting !expect/all node."
            scope_rewrites = expected_scope_rewrites[expect]
            for line_range in sorted(
                scope_rewrites.keys(), key=lambda lines: lines or (0, 0)
            ):
                var_expected_values = scope_rewrites[line_range]
                # First we determine which node will be the parent for the new expect nodes; then we can start appending
                # new expects to that parent's child list.
                if line_range is None:
                    new_expect_sibling_list = scope_where_children
                else:
                    start, stop = line_range
                    lines = (
                        Line(start)
                        if start == stop
                        else DexRange(Line(start), Line(stop))
                    )
                    new_expect_parent = Where({"lines": lines}, is_and=True)
                    # Reuse an existing !and node if one exists...
                    try:
                        existing_parent = next(
                            node
                            for node in scope_where_children
                            if str(node) == str(new_expect_parent)
                        )
                        new_expect_sibling_list = new_node_child_map.setdefault(
                            existing_parent, []
                        )
                    except StopIteration:
                        scope_where_children.append(new_expect_parent)
                        new_expect_sibling_list = new_node_child_map.setdefault(
                            new_expect_parent, []
                        )
                for var, expected_values in var_expected_values:
                    new_expect = expect.get_base_expect(var)
                    new_expect_sibling_list.append(new_expect)
                    new_node_child_map[new_expect] = expected_values
            return
        assert isinstance(expect, (Step, Type, Value))
        new_expected_value = add_expected_values.get(expect) or expected_value
        new_node_child_map[expect] = new_expected_value
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
