# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger output to script expected values."""

from collections import Counter, OrderedDict
import copy
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from dex.dextIR import ValueIR
from dex.test_script.Nodes import Expect, Address, Float


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
    elif isinstance(expected, Float):
        # Float nodes may themselves contain lists of values; we treat each of those as individual expected values.
        for expected_float in expected.get_expected_values():
            result[prepend_tuple + (expected_float,)] += 1
    else:
        result[prepend_tuple + (str(expected),)] += 1
    return result


class ExpectMatchContext:
    """Context class used to track evaluation state across variables/steps. Updated as new matches are made; since we
    try many matches and select the best one, we avoid committing any updates to this context until we have selected
    the final match."""

    def __init__(self):
        self.address_label_resolutions: Dict[str, int] = {}

    def commit(self, other: "ExpectMatchContext"):
        assert all(
            other.address_label_resolutions.get(addr)
            == self.address_label_resolutions[addr]
            for addr in self.address_label_resolutions
        ), "New committed address resolutions override existing resolutions!"
        self.address_label_resolutions = other.address_label_resolutions


class MatchResult(IntEnum):
    FALSE = 0
    PARTIAL = 1
    TRUE = 2

    @staticmethod
    def from_bools(is_true: bool, is_false: Optional[bool] = None) -> "MatchResult":
        """Returns a MatchResult based on the provided boolean value(s):
        - The single argument case simply returns TRUE if the argument is True, and FALSE otherwise.
        - The two argument case combines its arguments, giving TRUE if `is_true and not is_false`, FALSE for the
          inverse, and PARTIAL if `is_true and is_false`. Currently rejects `not is_true and not is_false`, as we don't
          intend to represent this state with a MatchResult.
        """
        if is_false is None:
            is_false = not is_true
        if is_true and not is_false:
            return MatchResult.TRUE
        if is_false and not is_true:
            return MatchResult.FALSE
        assert (
            is_false and is_true
        ), "Invalid inputs to MatchResult; cannot be not false and not true."
        return MatchResult.PARTIAL


class DebuggerExpectMatch:
    """Class that represents the match between a particular expected value for an Expect node and the actual debugger
    output corresponding to the watched value for that node.
    The value of `expected` may be either a string-convertible scalar (non-collection) value, representing a
    non-aggregate expected value, or it may be a dict of strings to other expected values.
    `actual_result` is None if either `actual` or `expect.get_variable_result(actual)` is None,
    Otherwise, if `expected` is a dict, then `actual_result` is a dict[str, DebuggerExpectMatch],
    Otherwise, `actual_result` is a str.
    Uses the provided match_context, and updates a local copy of it; if this match is selected, then its local updated
    match_context should be committed.
    """

    def __init__(
        self,
        expect: Expect,
        expected,
        actual: Optional[ValueIR],
        match_context: ExpectMatchContext,
    ):
        self.expect = expect
        self.expected = expected
        self.actual = actual
        # Create a local "provisional" copy of the match context. We may update this local context without affecting the
        # actual global match context, before this match is selected as the canonical match for the current expect+step.
        # If this match is selected, then we will commit any changes made in this provisional match_context back to the
        # global context.
        self.provisional_match_context = copy.deepcopy(match_context)
        self.actual_result, self.match_result = self._get_actual_result()
        self.match_distance = self._get_match_distance()

    def _get_actual_result(
        self,
    ) -> Tuple[Union[str, Dict[str, "DebuggerExpectMatch"], None], MatchResult]:
        if isinstance(self.expected, dict):
            return self._get_dict_actual_result(self.expected)

        actual_result = (
            self.expect.get_variable_result(self.actual)
            if self.actual is not None
            else None
        )
        if self.expected is None or actual_result is None:
            return actual_result, MatchResult.FALSE
        if isinstance(self.expected, Address):
            return self._get_address_actual_result(self.expected, actual_result)

        if isinstance(self.expected, Float):
            matched_expected = self.expected.matches(actual_result)
            if matched_expected is None:
                return actual_result, MatchResult.FALSE
            self.expected = matched_expected
            return actual_result, MatchResult.TRUE

        match_result = MatchResult.from_bools(str(self.expected) == actual_result)
        return actual_result, match_result

    def _get_address_actual_result(
        self, expected: Address, actual_result: str
    ) -> Tuple[Union[str, Dict[str, "DebuggerExpectMatch"], None], MatchResult]:
        """Returns the actual result for an !address expected value."""
        # First check whether the actual value we have is an address.
        try:
            actual_addr = int(actual_result.split(maxsplit=1)[0], 16)
        except ValueError:
            # Not a valid address, so we can't match.
            return actual_result, MatchResult.FALSE
        # If the address is already resolved, we just have to see if it matches.
        if (
            resolved_addr := self.provisional_match_context.address_label_resolutions.get(
                expected.name
            )
        ) is not None:
            return actual_result, MatchResult.from_bools(
                resolved_addr + expected.offset == actual_addr
            )
        # If the address is not resolved, then we can assign to it now in our local copy.
        resolved_addr = actual_addr - expected.offset
        self.provisional_match_context.address_label_resolutions[
            expected.name
        ] = resolved_addr
        return actual_result, MatchResult.TRUE

    def _get_dict_actual_result(
        self, expected: dict
    ) -> Tuple[Union[str, Dict[str, "DebuggerExpectMatch"], None], MatchResult]:
        """Returns the actual result for a 'dict' expected value."""
        sub_expect_results: Dict[str, DebuggerExpectMatch] = OrderedDict()
        for sub_expect, sub_expected in expected.items():
            # If the value of `actual` is None, we still want this match to reflect the structure of the expected
            # value, so if we have an expected value: `!value foo: {a: 0, b: 1}`, and `actual == None`, then we
            # should produce a match `foo: {'a': None, 'b': None}`, rather than `foo: None`, so we unconditionally
            # traverse the expected value here tree even if we have a None result.
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
            if value is None:
                sub_expect_results[sub_expect] = DebuggerExpectMatch(
                    self.expect, None, None, self.provisional_match_context
                )
            else:
                # Recursively invoke get_expect_match, which will create a `DebuggerExpectMatch` for all values in
                # sub_expected and select the best match, or return a match against None if sub_expected contains no
                # values that match `value`.
                sub_expect_results[sub_expect] = get_expect_match(
                    self.expect, sub_expected, value, self.provisional_match_context
                )
        match_result = MatchResult.from_bools(
            any(
                result.match_result == MatchResult.TRUE
                for result in sub_expect_results.values()
            ),
            any(
                result.match_result == MatchResult.FALSE
                for result in sub_expect_results.values()
            ),
        )
        return sub_expect_results, match_result

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


def get_expect_match(
    expect: Expect, expected_values, actual: ValueIR, match_context: ExpectMatchContext
):
    """Given one or more expected values for an Expect node and an actual ValueIR, returns a match for the best
    matching expected value, which is either the first exact match, or the match with the lowest distance (see
    `DebuggerExpectMatch._get_match_distance` above), or returns a match for None if there are no expected values with
    a match distance less than 1.0 (i.e. all expected values have no overlap with the actual value).
    """
    if not isinstance(expected_values, list):
        expected_values = [expected_values]
    best_match = DebuggerExpectMatch(expect, None, actual, match_context)
    best_match_dist = 1.0
    for expected_value in expected_values:
        expect_match = DebuggerExpectMatch(
            expect, expected_value, actual, match_context
        )
        if expect_match.match_result == MatchResult.TRUE:
            best_match = expect_match
            break
        # A "FALSE" match  will have a match distance of 1.0, and therefore will never be considered a "best match".
        if expect_match.match_distance < best_match_dist:
            best_match = expect_match
            best_match_dist = expect_match.match_distance

    match_context.commit(best_match.provisional_match_context)
    return best_match
