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
from typing import Any, Dict, List, Tuple

from dex.dextIR import DextIR, StepIR
from dex.evaluation.ExpectMatch import (
    DebuggerExpectMatch,
    MatchResult,
    get_expect_match,
)
from dex.evaluation.Metrics import (
    Metric,
    get_variable_metrics,
    serialize_metric_to_json,
)
from dex.evaluation.StateMatch import get_active_where_matches
from dex.test_script import DexterScript, Scope
from dex.test_script.Nodes import Expect, Value


class DebuggerStepMatch:
    """Class used to record the match between a DexterScript and a StepIR, including the state match, determining which
    script nodes are "active", and the expect matches, which compare the debugger's output to the DexterScript's
    expected output."""

    def __init__(self, step: StepIR, script: DexterScript):
        self.step = step
        self.script = script
        self.state_match = get_active_where_matches(script, step)
        expects_to_match = {
            expect
            for where_match in self.state_match.values()
            for expect in where_match.active_expects
        }
        self.expect_matches: Dict[Expect, DebuggerExpectMatch] = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            assert isinstance(expect, Value), "Non-Value expects currently unsupported"
            if expect in expects_to_match:
                self.expect_matches[expect] = get_expect_match(
                    expect, expected_value, step.watches[expect.get_watched_expr()]
                )

        script.visit_script(visit_expect=add_expected_values)


class DebuggerRunMatch(object):
    """Class used to record the complete match of a debugger session and a DexterScript. It is necessary to match
    step-by-step rather than variable-by-variable (i.e. we evaluate all variables for a step before the evaluating the
    next step), because there are features (yet to be implemented) which allow the match of one variable at step N to
    affect the match of another variable at step N+1, thus we go one step at a time.
    """

    def __init__(self, context, dext_ir: DextIR):
        self.context = context
        self.dext_ir = dext_ir
        self.metrics: Dict[str, Metric] = {}
        self.step_matches: List[DebuggerStepMatch] = []
        self.per_expect_results: Dict[
            Expect, list[Tuple[int, DebuggerExpectMatch]]
        ] = {}

        script = self.dext_ir.script
        assert script is not None, "Trying to evaluate DextIR without attached script?"

        # Gather the expected values for each Expect.
        expected_values = {}

        def add_expected_values(expect: Expect, expected_value: Any, scope: Scope):
            assert isinstance(expect, Value), "Non-Value expects currently unsupported"
            expected_values[expect] = expected_value
            self.per_expect_results[expect] = []

        script.visit_script(visit_expect=add_expected_values)

        # Then produce all of our step matches.
        for step in self.dext_ir.steps:
            self.step_matches.append(DebuggerStepMatch(step, script))

        # Then, for each expect, produce the list of results for just that variable.
        for step_match in self.step_matches:
            for expect, expect_match in step_match.expect_matches.items():
                self.per_expect_results[expect].append(
                    (step_match.step.step_index, expect_match)
                )

        # Finally, compare the match results against the expected values to produce the metrics.
        for expect, expect_results in self.per_expect_results.items():
            expect_matches = [match for step, match in expect_results]
            expect_metrics = get_variable_metrics(
                expect, expected_values[expect], expect_matches
            )
            for metric_name, metric in expect_metrics.items():
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
            for where, where_match in step_match.state_match.items():
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
            if not step_match.expect_matches:
                continue
            result += f"  Active !expect nodes:\n"
            matching_expects = [
                (expect, match)
                for expect, match in step_match.expect_matches.items()
                if match.match_result == MatchResult.TRUE
            ]
            non_matching_expects = [
                (expect, match)
                for expect, match in step_match.expect_matches.items()
                if match.match_result != MatchResult.TRUE
            ]
            if matching_expects:
                result += f"    Matching nodes:     [{', '.join(f'{expect}={match.short_str()}' for expect, match in matching_expects)}]\n"
            if non_matching_expects:
                result += f"    Non-matching nodes: [{', '.join(f'{expect}={match.short_str()}' for expect, match in non_matching_expects)}]\n"
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
