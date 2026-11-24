# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Produce metric results from the results of a comparison of a DexterScript and debugger output.
"""

from typing import Any, Dict, List, Optional, Union

from dex.evaluation.ExpectMatch import (
    DebuggerExpectMatch,
    MatchResult,
    get_expected_value_set,
)
from dex.test_script.Nodes import Expect, Step, Type, Value


class Metric:
    def __init__(self, improves_asc=True):
        self.improves_asc = improves_asc

    def as_scalar(self) -> float:
        raise NotImplementedError()

    def aggregate(self, other):
        raise NotImplementedError()

    # Returns 1 if this metric is better than "other", -1 if it worse, and 0 if it is the same.
    def compare(self, other):
        a = self.as_scalar()
        b = other.as_scalar()
        if not self.improves_asc:
            a, b = b, a
        if a > b:
            return 1
        elif a < b:
            return -1
        else:
            return 0


class ScalarMetric(Metric):
    def __init__(self, value: Union[int, float], improves_asc=True):
        self.value = value
        super().__init__(improves_asc)

    def as_scalar(self) -> float:
        return float(self.value)

    def aggregate(self, other):
        assert (
            self.improves_asc == other.improves_asc
        ), "Trying to aggregate different metrics?"
        return ScalarMetric(self.value + other.value, self.improves_asc)

    def __repr__(self):
        if isinstance(self.value, float):
            return f"{self.value:.4}"
        return f"{self.value}"


class FractionMetric(Metric):
    def __init__(self, numerator: int, denominator: int, improves_asc=True):
        self.num = numerator
        self.dom = denominator
        super().__init__(improves_asc)

    def as_scalar(self) -> float:
        if self.dom == 0:
            return float("nan")
        return float(self.num) / float(self.dom)

    def as_pct(self) -> float:
        return self.as_scalar() * 100

    def aggregate(self, other):
        assert (
            self.improves_asc == other.improves_asc
        ), "Trying to aggregate different metrics?"
        return FractionMetric(
            self.num + other.num, self.dom + other.dom, self.improves_asc
        )

    def __repr__(self):
        return f"{self.as_pct():.1f}% ({self.num}/{self.dom})"


def serialize_metric_to_json(metric):
    if isinstance(metric, ScalarMetric):
        return metric.value
    elif isinstance(metric, FractionMetric):
        return metric.as_pct()
    raise Exception("Invalid metric type!")


def get_variable_metrics(
    expect: Expect, expected_values: Any, matches: List[DebuggerExpectMatch]
) -> Dict[str, Metric]:
    """Given an Expect node with its expected values and a list of all matches for that Expect in a debugger session,
    returns the computed metrics for that Expect node."""
    assert isinstance(expect, (Type, Value)), f"Unexpected non-variable expect {expect}"
    if not isinstance(expected_values, list):
        expected_values = [expected_values]
    num_total_steps = len(matches)
    all_expected_values = get_expected_value_set(expected_values)
    seen_expected_values = set()
    num_correct_steps = 0
    num_optimized_out_steps = 0
    num_irretrievable_steps = 0
    num_missing_var_steps = 0
    num_unexpected_value_steps = 0
    partial_step_correctness = 0.0
    for match in matches:
        partial_step_correctness += 1.0 - match.match_distance
        seen_expected_values = seen_expected_values.union(
            match.get_all_matched_values()
        )
        if match.match_result == MatchResult.TRUE:
            num_correct_steps += 1
        elif match.actual_result is None:
            if match.actual and match.actual.is_optimized_away:
                num_optimized_out_steps += 1
            elif match.actual and match.actual.is_irretrievable:
                num_irretrievable_steps += 1
            else:
                num_missing_var_steps += 1
        else:
            num_unexpected_value_steps += 1
    assert all(
        ev in all_expected_values for ev in seen_expected_values
    ), "Saw expected values that weren't expected?"
    num_seen_values = sum(all_expected_values[ev] for ev in seen_expected_values)
    num_missing_values = sum(
        0 if ev in seen_expected_values else count
        for ev, count in all_expected_values.items()
    )
    kind_string = "value" if isinstance(expect, Value) else "type"
    # And finally produce the metrics map and add the new result to the list.
    metrics = {
        # The number of steps. Though this is not a useful metric in itself, it may be useful to see in tandem with
        # other variables.
        "total_watched_steps": ScalarMetric(num_total_steps),
        # The number of steps where the expected value sequence was observed.
        "correct_steps": ScalarMetric(num_correct_steps),
        # The number of steps which did not match the expected value sequence.
        "incorrect_steps": ScalarMetric(
            num_total_steps - num_correct_steps, improves_asc=False
        ),
        # The sum of the 0.0-1.0 "correctness value" of matches across each step.
        "partial_step_correctness": ScalarMetric(partial_step_correctness),
        # The number of steps where the watched variable/expression was marked "optimized out" in the debugger.
        "optimized_out_steps": ScalarMetric(
            num_optimized_out_steps, improves_asc=False
        ),
        # The number of steps where the watched variable/expression had an inaccessible address in the debugger.
        "irretrievable_steps": ScalarMetric(
            num_irretrievable_steps, improves_asc=False
        ),
        # The number of steps where the watched variable/expression was not available in the debugger.
        "missing_var_steps": ScalarMetric(num_missing_var_steps, improves_asc=False),
        # The number of steps where the watched variable/expression had a value not in the set of expected values.
        f"unexpected_{kind_string}_steps": ScalarMetric(
            num_unexpected_value_steps, improves_asc=False
        ),
        # The % of steps where the expected value sequence was observed.
        "correct_step_coverage": FractionMetric(num_correct_steps, num_total_steps),
        # The number of expected values that were observed at least once.
        f"seen_{kind_string}s": ScalarMetric(num_seen_values),
        # The number of expected values that were not observed.
        f"missing_{kind_string}s": ScalarMetric(num_missing_values, improves_asc=False),
    }
    return metrics


def lcs_len(a: List[int], b: List[int]) -> int:
    """Returns the length of the longest common subsequence between a and b."""
    lcs_table: List[List[int]] = [
        [0 for _ in range(len(b) + 1)] for _ in range(len(a) + 1)
    ]
    for a_idx in range(len(a)):
        for b_idx in range(len(b)):
            if a[a_idx] == b[b_idx]:
                lcs_table[a_idx + 1][b_idx + 1] = 1 + lcs_table[a_idx][b_idx]
            else:
                lcs_table[a_idx + 1][b_idx + 1] = max(
                    lcs_table[a_idx + 1][b_idx], lcs_table[a_idx][b_idx + 1]
                )
    return lcs_table[-1][-1]


def get_step_metrics(
    expect: Step, expected_lines: List[int], step_lines: List[int]
) -> Dict[str, Metric]:
    """Given an Expect node with its expected values and a list of all matches for that Expect in a debugger session,
    returns the computed metrics for that Expect node."""

    expected_line_set = set(expected_lines)
    actual_line_set = set(step_lines)

    total_line_steps = len(step_lines)
    if expect.kind == "exactly" or expect.kind == "at_least":
        num_matching_steps = lcs_len(expected_lines, step_lines)
        # Inefficient, but not to the point that we care!
        num_matching_steps_ignoring_order = lcs_len(
            sorted(expected_lines), sorted(step_lines)
        )

        max_possible_correct_line_steps = len(expected_lines)
        correct_line_steps = num_matching_steps
        misordered_line_steps = num_matching_steps_ignoring_order - num_matching_steps
        missing_lines = len(expected_line_set - actual_line_set)
        if expect.kind == "exactly":
            incorrect_line_steps = total_line_steps - correct_line_steps
            unexpected_lines = len(actual_line_set - expected_line_set)
        else:
            # For `!step at_least` there are no "incorrect" or "unexpected" lines, since we explicitly ignore seen lines
            # outside of the expected lines.
            incorrect_line_steps = 0
            unexpected_lines = 0
    else:
        assert expect.kind == "never"
        max_possible_correct_line_steps = total_line_steps
        correct_line_steps = sum(
            1 for line in step_lines if line not in expected_line_set
        )
        incorrect_line_steps = total_line_steps - correct_line_steps
        unexpected_lines = len(actual_line_set.intersection(expected_line_set))
        # For `!step never` there are no "missing" or "misordered" lines, since we only declare lines we *don't* want to
        # see.
        missing_lines = 0
        misordered_line_steps = 0

    metrics: Dict[str, Metric] = {
        "total_line_steps": ScalarMetric(total_line_steps),
        "correct_line_steps": ScalarMetric(correct_line_steps),
        "correct_line_score": FractionMetric(
            correct_line_steps, max_possible_correct_line_steps
        ),
        "misordered_line_steps": ScalarMetric(
            misordered_line_steps, improves_asc=False
        ),
        "missing_lines": ScalarMetric(missing_lines, improves_asc=False),
        "incorrect_line_steps": ScalarMetric(incorrect_line_steps, improves_asc=False),
        "unexpected_lines": ScalarMetric(unexpected_lines, improves_asc=False),
    }

    return metrics
