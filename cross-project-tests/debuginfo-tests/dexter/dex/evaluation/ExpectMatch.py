# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger output to script expected values."""

from collections import Counter, OrderedDict
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dex.dextIR import ValueIR
from dex.test_script.Nodes import Expect, Value


def get_expected_value_set(
    expected, prepend_tuple: Tuple = ()
) -> Dict[Tuple[str], int]:
    """For the given "expected" taken directly from the script YAML, returns the set of all actual expected
    values, using tuples to represent nested expected values, and mapping each result to the number of times that it
    appears in the set (giving a count). For example, the expected values of:
    ```
    !value foo:
    - 4
    - 6
    - x: 5
      y: 10
    - x: 5
      y: 20
    ```
    Would be represented by the dict:
    ```
    {
      (4,): 1,
      (6,): 1,
      ("x", 5): 2,
      ("y", 10): 1,
      ("y", 20): 1,
    }
    ```
    """
    result: Dict[Tuple, int] = Counter()
    if isinstance(expected, list):
        for ev in expected:
            result.update(get_expected_value_set(ev, prepend_tuple))
    elif isinstance(expected, dict):
        for sub_expect, sub_expected in expected.items():
            next_prepend = prepend_tuple + (str(sub_expect),)
            result.update(get_expected_value_set(sub_expected, next_prepend))
    else:
        result[prepend_tuple + (str(expected),)] += 1
    return result


class MatchResult(IntEnum):
    FALSE = 0
    PARTIAL = 1
    TRUE = 2


class DebuggerExpectMatch:
    """Class that represents the match between a particular expected value for an Expect node and the actual debugger
    output corresponding to the watched value for that node.
    `actual_result` is None if `actual` or `expect.get_variable_result(actual)` is None,
    Otherwise, if `expected` is a dict, then `actual_result` is a dict[str, DebuggerExpectMatch],
    Otherwise, `actual_result` is a str.
    """

    def __init__(self, expect: Expect, expected, actual: Optional[ValueIR]):
        self.expect = expect
        self.expected = expected
        self.actual = actual
        self.actual_result, self.match_result = self._get_actual_result()
        self.match_distance = self._get_match_distance()

    def _get_actual_result(
        self,
    ) -> Tuple[Union[str, Dict[str, "DebuggerExpectMatch"], None], MatchResult]:
        if isinstance(self.expected, dict):
            sub_expect_results: Dict[str, DebuggerExpectMatch] = OrderedDict()
            for sub_expect, sub_expected in self.expected.items():
                value = (
                    None
                    if self.actual is None
                    else next(
                        (
                            sub_value
                            for sub_value in self.actual.sub_values
                            if sub_value.expression == sub_expect
                        ),
                        None,
                    )
                )
                sub_expect_results[sub_expect] = DebuggerExpectMatch(
                    self.expect, sub_expected, value
                )
            if all(
                result.match_result == MatchResult.TRUE
                for result in sub_expect_results.values()
            ):
                match_result = MatchResult.TRUE
            elif all(
                result.match_result == MatchResult.FALSE
                for result in sub_expect_results.values()
            ):
                match_result = MatchResult.FALSE
            else:
                match_result = MatchResult.PARTIAL
            return sub_expect_results, match_result

        actual_result = (
            self.expect.get_variable_result(self.actual)
            if self.actual is not None
            else None
        )
        match_result = (
            MatchResult.TRUE
            if (self.expected is not None and str(self.expected) == actual_result)
            else MatchResult.FALSE
        )
        return actual_result, match_result

    def _get_match_distance(self) -> float:
        if self.match_result == MatchResult.TRUE:
            return 0.0
        if self.match_result == MatchResult.FALSE:
            return 1.0
        assert (
            isinstance(self.actual_result, Dict) and self.actual_result
        ), "Partial match without submatches."
        dists = [m.match_distance for m in self.actual_result.values()]
        return sum(dists) / len(dists)

    def get_expression(self) -> Optional[str]:
        return self.actual.expression if self.actual else None

    def get_all_matched_values(self, prepend_tuple: Tuple = ()) -> Set[Tuple]:
        """Similar to `get_expected_value_set` above, but returns the set of expected values that were successfully
        matched in this DebuggerExpectMatch, and returns just a set (no counts)."""
        if self.match_result == MatchResult.FALSE:
            return set()
        assert (
            self.actual_result is not None and self.actual is not None
        ), "Non-false match with no actual result."
        if isinstance(self.actual_result, str):
            assert (
                self.match_result == MatchResult.TRUE
            ), "Partial match without submatches."
            return {prepend_tuple + (str(self.expected),)}
        result = set()
        for sub_expr, sub_result in self.actual_result.items():
            result = result.union(
                sub_result.get_all_matched_values(prepend_tuple + (sub_expr,))
            )
        return result

    def short_str(self, use_color=True) -> str:
        def colorize(input: str, match_result: MatchResult) -> str:
            if not use_color:
                return input
            if match_result == MatchResult.TRUE:
                return f"<g>{input}</>"
            if match_result == MatchResult.PARTIAL:
                return f"<y>{input}</>"
            return f"<r>{input}</>"

        if self.actual is None:
            return colorize("<Missing>", self.match_result)
        if self.actual_result is None:
            if self.actual.is_optimized_away:
                return colorize("<OptimizedOut>", self.match_result)
            if self.actual.is_irretrievable:
                return colorize("<Irretrievable>", self.match_result)
            return colorize("<EvaluateFailed>", self.match_result)
        if isinstance(self.actual_result, str):
            return colorize(self.actual_result, self.match_result)
        assert isinstance(self.expected, dict)
        sub_values = [
            colorize(f'"{sub_expr}": ', sub_result.match_result)
            + sub_result.short_str()
            for sub_expr, sub_result in self.actual_result.items()
        ]
        return f"{{ {', '.join(sub_values)} }}"


def get_expect_match(expect: Expect, expected_values, actual: ValueIR):
    """Given one or more expected values for an Expect node and an actual ValueIR, returns a match for the first
    matching expected values, or for None if there are no matching expected values."""
    if not isinstance(expected_values, list):
        expected_values = [expected_values]
    best_partial_match = DebuggerExpectMatch(expect, None, actual)
    best_partial_match_dist = 1.0
    for expected_value in expected_values:
        expect_match = DebuggerExpectMatch(expect, expected_value, actual)
        if expect_match.match_result == MatchResult.TRUE:
            return expect_match
        # A "FALSE" match  will have a match distance of 1.0, and therefore will never be considered a "best match".
        if expect_match.match_distance < best_partial_match_dist:
            best_partial_match = expect_match
            best_partial_match_dist = expect_match.match_distance

    return best_partial_match
