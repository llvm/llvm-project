# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes for matching observed debugger output to script expectations.
"""

# For each command, there is a set of metrics that can be generated. Metrics across multiple identical commands can be
# aggregated, and each individual metric can be expressed in a scalar form that is considered "better" as it either
# ascends or descends.
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from dex.dextIR import DextIR, StepIR
from dex.evaluation.ExpectMatch import (
    DebuggerExpectMatch,
    ExpectMatchContext,
    MatchResult,
    get_expect_match,
)
from dex.evaluation.Metrics import (
    Metric,
    get_step_metrics,
    get_variable_metrics,
    serialize_metric_to_json,
)
from dex.evaluation.StateMatch import StateMatchContext, get_state_match
from dex.test_script import DexterScript, Scope
from dex.test_script.Nodes import Expect, ExpectAll, Line, Step

class DebuggerStepMatch:
    """Class used to record the match between a DexterScript and a StepIR, including the state match, determining which
    script nodes are "active", and the expect matches, which compare the debugger's output to the DexterScript's
    expected output."""

    def __init__(
        self,
        step: StepIR,
        script: DexterScript,
        match_context: ExpectMatchContext,
        state_match_context: StateMatchContext,
    ):
        self.step = step
        self.script = script
        self.match_context = match_context
        self.state_match = get_state_match(script, step, state_match_context)
        expects_to_match = {
            expect: where_match.frame_idx
            for where_match in self.state_match.where_match_results.values()
            for expect in where_match.active_expects
        }
        self.var_expect_matches: Dict[Expect, DebuggerExpectMatch] = {}
        self.step_expect_matches: Dict[Step, int] = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            if expect not in expects_to_match:
                return
            expect_frame_idx = expects_to_match[expect]
            if isinstance(expect, Step):
                self.step_expect_matches[expect] = step.frames[
                    expect_frame_idx
                ].loc.lineno
                return
            assert (
                expect.get_watched_expr() is not None
            ), f"Unexpected expect node kind {expect}"
            self.var_expect_matches[expect] = get_expect_match(
                expect,
                expected_value,
                step.frames[expect_frame_idx].watches[expect.get_watched_expr()],
                self.match_context,
            )

        script.visit_script(visit_expect=add_expected_values)


class DebuggerRunMatch(object):
    """Class used to record the complete match of a debugger session and a DexterScript. It is necessary to match
    step-by-step rather than variable-by-variable (i.e. we evaluate all variables for a step before the evaluating the
    next step), because there are features (yet to be implemented) which allow the match of one variable at step N to
    affect the match of another variable at step N+1, thus we go one step at a time.
    """

    def __init__(self, dex_context, dext_ir: DextIR):
        self.dex_context = dex_context
        self.match_context = ExpectMatchContext()
        self.dext_ir = dext_ir
        self.metrics: Dict[str, Metric] = {}
        self.step_matches: List[DebuggerStepMatch] = []
        self.per_var_expect_results: Dict[
            Expect, list[Tuple[int, DebuggerExpectMatch]]
        ] = {}
        self.per_step_expect_results: Dict[Step, list[Tuple[int, int]]] = {}

        script = self.dext_ir.script
        assert script is not None, "Trying to evaluate DextIR without attached script?"

        # Gather the expected values for each Expect.
        self.expected_values = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            self.expected_values[expect] = expected_value
            if expect.get_watched_expr() is not None:
                self.per_var_expect_results[expect] = []
                return
            assert isinstance(expect, Step), f"Unexpected expect node kind {expect}"
            self.per_step_expect_results[expect] = []

        script.visit_script(visit_expect=add_expected_values)

        # Then produce all of our step matches.
        def check_condition(step: StepIR, frame_idx: int, condition: str):
            cond_value = step.frames[frame_idx].watches[condition]
            result = cond_value.could_evaluate and cond_value.value.lower() == "true"
            return result

        state_match_context = StateMatchContext(check_condition=check_condition)
        for step in self.dext_ir.steps:
            self.step_matches.append(
                DebuggerStepMatch(step, script, self.match_context, state_match_context)
            )

        # Then, for each expect, produce the list of results for just that variable.
        for step_match in self.step_matches:
            for step_expect, line in step_match.step_expect_matches.items():
                self.per_step_expect_results[step_expect].append(
                    (step_match.step.step_index, line)
                )
            for expect, expect_match in step_match.var_expect_matches.items():
                self.per_var_expect_results[expect].append(
                    (step_match.step.step_index, expect_match)
                )

        # For !steps, once we know the file that they are in, we apply any labels.
        for step_expect, step_results in self.per_step_expect_results.items():
            if not step_results:
                # We may not be able to resolve any !labels in the expected value list if the Expect was never active;
                # as a workaround, just set any integers here - the result will be the same.
                self.expected_values[step_expect] = [
                    0 for l in self.expected_values[step_expect]
                ]
                continue
            active_path = self.dext_ir.steps[0].current_location.path
            assert all(
                self.dext_ir.steps[step_index].current_location.path == active_path
                for step_index, line in step_results[1:]
            ), "!step node unexpectedly active over multiple files"
            path_labels = script.get_labels(active_path)
            self.expected_values[step_expect] = [
                Line(l).to_line(path_labels) for l in self.expected_values[step_expect]
            ]

        # Finally, compare the match results against the expected values to produce the metrics.
        for expect, expect_results in self.per_var_expect_results.items():
            expect_matches = [match for step, match in expect_results]
            expect_metrics = get_variable_metrics(
                expect, self.expected_values[expect], expect_matches
            )
            for metric_name, metric in expect_metrics.items():
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = metric
                else:
                    self.metrics[metric_name] = self.metrics[metric_name].aggregate(
                        metric
                    )
        for expect, lines in self.per_step_expect_results.items():
            actual_lines = [lines for step, lines in lines]
            step_metrics = get_step_metrics(
                expect, self.expected_values[expect], actual_lines
            )
            for metric_name, metric in step_metrics.items():
                if metric_name not in self.metrics:
                    self.metrics[metric_name] = metric
                else:
                    self.metrics[metric_name] = self.metrics[metric_name].aggregate(
                        metric
                    )

    def dump_step_results(self) -> str:
        result = ""
        for step_match in self.step_matches:
            result += f"Step {step_match.step.step_index}:\n"
            result += f"  {step_match.step.current_location}\n"
            frame_active_wheres = defaultdict(list)
            for (
                where,
                where_match,
            ) in step_match.state_match.where_match_results.items():
                frame_active_wheres[where_match.frame_idx].append(str(where))
            if not frame_active_wheres:
                result += f"  No active !where nodes.\n"
                continue
            frame_active_wheres_list = sorted(
                [
                    (frame_idx, wheres)
                    for frame_idx, wheres in frame_active_wheres.items()
                ],
                key=lambda entry: entry[0],
            )
            result += f"  Active !where nodes:\n"
            for frame_idx, wheres in frame_active_wheres_list:
                result += f"    Frame {frame_idx}: [{', '.join(wheres)}]\n"
            if not step_match.var_expect_matches and not step_match.step_expect_matches:
                continue
            result += f"  Active !expect nodes:\n"
            matching_expects = [
                (expect, match.short_str())
                for expect, match in step_match.var_expect_matches.items()
                if match.match_result == MatchResult.TRUE
            ]
            non_matching_expects = [
                (expect, match.short_str())
                for expect, match in step_match.var_expect_matches.items()
                if match.match_result != MatchResult.TRUE
            ]

            def step_expect_matches(expect: Step, step_line: int) -> bool:
                expected_lines = self.expected_values[expect]
                assert isinstance(expected_lines, list) and all(
                    isinstance(l, int) for l in expected_lines
                )
                step_line_in_expected_list = step_line in expected_lines
                if expect.kind == "never":
                    return not step_line_in_expected_list
                return step_line_in_expected_list

            for step_expect, step_line in step_match.step_expect_matches.items():
                list_to_append = (
                    matching_expects
                    if step_expect_matches(step_expect, step_line)
                    else non_matching_expects
                )
                list_to_append.append((step_expect, str(step_line)))
            if matching_expects:
                result += f"    Matching nodes:     [{', '.join(f'{expect}={match}' for expect, match in matching_expects)}]\n"
            if non_matching_expects:
                result += f"    Non-matching nodes: [{', '.join(f'{expect}={match}' for expect, match in non_matching_expects)}]\n"
        return result

    def get_metric_output(self):
        if not self.metrics:
            return "No expects found."
        lines = []
        for metric_type, metric in self.metrics.items():
            lines.append(f"{metric_type}: {metric}")
        return "\n".join(lines) + "\n"

    def get_metric_json_output(self):
        if not self.metrics:
            return "No expects found."
        return {
            metric_type: serialize_metric_to_json(metric)
            for metric_type, metric in self.metrics.items()
        }
