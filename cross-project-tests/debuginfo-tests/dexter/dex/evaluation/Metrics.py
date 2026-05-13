# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Produce metric results from the results of a comparison of a DexterScript and debugger output.
"""

from typing import Any, Dict, List, Union

from dex.evaluation.ExpectMatch import DebuggerExpectMatch
from dex.test_script.Nodes import Expect, Value


class Metric:
    def __init__(self, improves_asc = True):
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
    def __init__(self, value: Union[int, float], improves_asc = True):
        self.value = value
        super().__init__(improves_asc)

    def as_scalar(self) -> float:
        return float(self.value)

    def aggregate(self, other):
        return ScalarMetric(self.value + other.value, self.improves_asc)

    def __repr__(self):
        return f"{self.value}"

class FractionMetric(Metric):
    def __init__(self, numerator: int, denominator: int, improves_asc = True):
        self.num = numerator
        self.dom = denominator
        super().__init__(improves_asc)

    def as_scalar(self) -> float:
        return float(self.num) / float(self.dom)

    def as_pct(self) -> float:
        return self.as_scalar() * 100

    def aggregate(self, other):
        return FractionMetric(self.num + other.num, self.dom + other.dom, self.improves_asc)

    def __repr__(self):
        return f"{self.as_pct():.1f}% ({self.num}/{self.dom})"

def serialize_metric_to_json(metric):
    if isinstance(metric, ScalarMetric):
        return metric.value
    elif isinstance(metric, FractionMetric):
        return metric.as_pct()
    raise Exception("Invalid metric type!")

def get_variable_metrics(expect: Expect, expected_values: Any, matches: List[DebuggerExpectMatch]) -> Dict[str, Metric]:
    """Given an Expect node with its expected values and a list of all matches for that Expect in a debugger session,
    returns the computed metrics for that Expect node."""
    assert isinstance(expect, Value), "Non-Value expects currently unsupported"
    if not isinstance(expected_values, list):
        expected_values = [expected_values]
    num_total_steps = len(matches)
    seen_expected_values = set()
    num_correct_steps = 0
    num_missing_var_steps = 0
    num_unexpected_value_steps = 0
    for match in matches:
        if match.match_result:
            seen_expected_values.add(match.expected)
            num_correct_steps += 1
        elif match.actual_result is None:
            num_missing_var_steps += 1
        else:
            num_unexpected_value_steps += 1
    num_seen_values = sum(1 for ev in expected_values if ev in seen_expected_values)
    # And finally produce the metrics map and add the new result to the list.
    metrics = {
        # The number of steps. Though this is not a useful metric in itself, it may be useful to see in tandem with
        # other variables.
        "total_watched_steps": ScalarMetric(num_total_steps),
        # The number of steps where the expected value sequence was observed.
        "correct_steps": ScalarMetric(num_correct_steps),
        # The number of steps which did not match the expected value sequence.
        "incorrect_steps": ScalarMetric(num_total_steps - num_correct_steps, improves_asc=False),
        # The number of steps where the watched variable/expression was not available in the debugger.
        "missing_var_steps": ScalarMetric(num_missing_var_steps, improves_asc=False),
        # The number of steps where the watched variable/expression had a value not in the set of expected values.
        "unexpected_value_steps": ScalarMetric(
            num_unexpected_value_steps, improves_asc=False
        ),
        # The % of steps where the expected value sequence was observed.
        "correct_step_coverage": FractionMetric(num_correct_steps, num_total_steps),
        # The number of expected values that were observed at least once.
        "seen_values": ScalarMetric(num_seen_values),
        # The number of expected values that were not observed.
        "missing_values": ScalarMetric(len(expected_values) - num_seen_values, improves_asc=False),
    }
    return metrics
