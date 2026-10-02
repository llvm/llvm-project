# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger state, such as the call stack, conditions, or historical state (e.g. breakpoint
hitcounts) to descriptions of expected state in a DexterScript."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import os
from typing import Callable, Dict, List, Optional, Set

from dex.dextIR import FrameIR, StepIR
from dex.test_script import DexterScript, Scope
from dex.test_script.Nodes import Expect, FileLabels, Where, Then


class StateMatchContext:
    """Class that holds any state needed for matching state nodes to debugger state across a run."""

    def __init__(self, check_condition: Callable[[StepIR, int, str], bool]):
        self.where_hit_counts: Counter[Where] = Counter()
        self.expired_wheres: Set[Where] = set()
        self._last_match_result: Optional[StateMatchResult] = None
        # To avoid constantly re-evaluating conditions above the current function, and potentially causing them to
        # be unfulfillable if we have imperfect stack unwinding, we track conditions that have been found True for state
        # nodes above the current function and consider those conditions true until we return to/pass that frame.
        # Key is a frame index counting from the root upwards, to keep stable as we grow and shrink the stack.
        self._cached_frame_conditions: Dict[int, Dict[str, bool]] = defaultdict(dict)
        self._check_condition = check_condition

    def check_condition(self, step: StepIR, frame_idx: int, condition: str) -> bool:
        reverse_frame_idx = step.num_frames - frame_idx - 1
        cached_conditions = self._cached_frame_conditions[reverse_frame_idx]
        if condition in cached_conditions:
            return cached_conditions[condition]
        # In an ideal world we would always cache conditions in a caller frame before moving to a called frame, but
        # some optimized code makes this infeasible, so we settle for computing it after reaching the called frame
        # instead.
        result = self._check_condition(step, frame_idx, condition)
        # We cache this now, but we won't actually use it *unless* the next step adds a new frame (i.e. we step into a
        # call).
        self._cached_frame_conditions[reverse_frame_idx][condition] = result
        return result

    def refresh_condition_cache(self, step: StepIR):
        """Call once we start matching a new step, to clear out any stale/invalid cached conditions."""
        to_delete = []
        for reverse_frame_idx in self._cached_frame_conditions:
            # Any cached condition for the current frame, or a lower frame no longer on the stack at all, must be
            # cleared.
            if reverse_frame_idx + 1 >= step.num_frames:
                to_delete.append(reverse_frame_idx)
        for idx in to_delete:
            del self._cached_frame_conditions[idx]

    def where_hit_is_new(self, where: Where, step: StepIR) -> bool:
        """Returns True if the current step can be counted as a new "hit" for `where`, assuming that `where` was hit in
        this step (but does not check if `where` has actually been hit in the current step).
        """
        # If this is the first step, all hits are new.
        if self._last_match_result is None:
            return True
        # If this !where did not appear in any frame in the previous step, this is a fresh hit.
        if (
            where not in self._last_match_result.where_match_results
            and where not in self._last_match_result.early_wheres
        ):
            return True
        # If !where uses a function breakpoint and that breakpoint was hit this step, this is a fresh hit.
        if where.function and not where.lines and where in step.hit_where_bps:
            return True
        return False

    def add_hit_if_where_hit_is_new(self, where: Where, step: StepIR) -> bool:
        """Checks whether the current step can be counted as a new "hit" for `where`. Increments `where`'s hit count if
        it has a new hit, and returns True iff so."""
        assert (
            where.for_hit_count is not None or where.after_hit_count is not None
        ), "Tried to add hit count for !where without for/after_hit_count?"
        if self.where_hit_is_new(where, step):
            self.where_hit_counts[where] += 1
            print(f"Added hit count for {where}")
            if where.for_hit_count is not None and self.where_hit_counts[
                where
            ] >= where.for_hit_count + (where.after_hit_count or 0):
                self.expired_wheres.add(where)
            return True
        return False

    def update(self, new_match_result: "StateMatchResult"):
        self._last_match_result = new_match_result


def is_subpath(subpath: str, superpath: str) -> bool:
    """Returns True if subpath is a trailing subpath of superpath, i.e. if `superpath` ends with `subpath` after
    normalizing both paths."""
    normalized_subpath: str = os.path.normcase(os.path.normpath(subpath))
    normalized_superpath: str = os.path.normcase(os.path.normpath(superpath))
    return normalized_superpath.endswith(normalized_subpath)


class WhereFrameMatchResult(IntEnum):
    FALSE = 0
    TRUE = 1
    EARLY = 2


def _match_where_to_frame(
    where: Where,
    frame_idx: int,
    step: StepIR,
    labels: FileLabels,
    context: StateMatchContext,
    default_path: Optional[str] = None,
) -> WhereFrameMatchResult:
    """A very simple matcher, returns True iff `where` matches `frame`."""
    frame = step.frames[frame_idx]
    file = where.file
    if not file and where.lines and not where.function:
        file = default_path
    if file is not None and not is_subpath(file, frame.loc.path):
        return WhereFrameMatchResult.FALSE
    if where.function is not None:
        fn = frame.function
        if "(" in fn:
            fn = fn.split("(")[0]
        if where.function != fn:
            return WhereFrameMatchResult.FALSE
    if where.lines is not None:
        if frame.loc.lineno not in where.get_lines(labels):
            return WhereFrameMatchResult.FALSE
    if where.for_hit_count is not None:
        after_hit_count = where.after_hit_count or 0
        where_hit_count = context.where_hit_counts[where]
        if where_hit_count > where.for_hit_count + (after_hit_count):
            return WhereFrameMatchResult.FALSE
    # We place the condition check as far down as possible to avoid unnecessary debugger calls.
    if where.conditions is not None:
        if not context.check_condition(step, frame_idx, where.conditions):
            return WhereFrameMatchResult.FALSE
    # The check for after_hit_count must go last, as before we return EARLY, we need to know that the only condition
    # preventing the match is after_hit_count.
    if where.after_hit_count is not None:
        where_hit_count = context.where_hit_counts[where]
        if where_hit_count <= where.after_hit_count:
            return WhereFrameMatchResult.EARLY
    return WhereFrameMatchResult.TRUE


def match_where_to_frame(
    where: Where,
    frame_idx: int,
    step: StepIR,
    labels: FileLabels,
    context: StateMatchContext,
    default_path: Optional[str] = None,
) -> WhereFrameMatchResult:
    """Returns True if `where` matches `frame`. As part of this check, we perform the where-to-frame check once, and if
    we get a result that could change due to an increased hit count (i.e. if we get a match with a `for_hit_count`
    where, or if we get an "early" result), we increment `where`'s hit count and run the check again to check whether
    the result changes."""
    result = _match_where_to_frame(
        where, frame_idx, step, labels, context, default_path
    )
    if result == WhereFrameMatchResult.EARLY or (
        result == WhereFrameMatchResult.TRUE and where.for_hit_count is not None
    ):
        if context.add_hit_if_where_hit_is_new(where, step):
            result = _match_where_to_frame(
                where, frame_idx, step, labels, context, default_path
            )
    return result


@dataclass
class WhereMatchResult:
    """Class storing the result of a single !where matched against a stack frame. The primary information stored is just
    the frame index that the !where matched against; for convenience, the children of the !where are also included.
    """

    frame_idx: int
    active_expects: List[Expect] = field(default_factory=list)
    active_thens: List[Then] = field(default_factory=list)
    pending_wheres: List[Where] = field(default_factory=list)
    expired_wheres: List[Where] = field(default_factory=list)
    early_wheres: List[Where] = field(default_factory=list)


class StateMatchResult:
    def __init__(
        self,
        where_match_results: Dict[Where, WhereMatchResult],
        early_wheres: Set[Where],
    ):
        self.where_match_results = where_match_results
        self.early_wheres = early_wheres


def get_state_match(
    script: DexterScript, step_info: StepIR, match_context: StateMatchContext
) -> StateMatchResult:
    """Match the script against the step_info, producing a dict that maps each !where that matches a stack frame to the
    index of the (rootmost) stack frame that it matches, and if the frame that it matches is the current stack frame
    (i.e. the frame index is 0), also includes a list of every direct child !expect node for that !where.
    """
    active_where_expects: Dict[Where, WhereMatchResult] = {}
    early_wheres: Set[Where] = set()
    match_context.refresh_condition_cache(step_info)

    def get_active_wheres(where: Where, scope: Scope):
        # For nested !wheres, we must match a specific frame relative to the parent !where.
        expected_file = scope.get_known_file_for_where(where)
        if scope.where:
            if scope.where not in active_where_expects:
                # If the parent !where doesn't match any frame, then this !where cannot match any either.
                return
            parent_frame_idx = active_where_expects[scope.where].frame_idx
            # !and must match the same frame as its parent; !where must match the next leafmost frame.
            target_frame_idx = (
                parent_frame_idx if where.is_and else parent_frame_idx - 1
            )
            if target_frame_idx < 0:
                # If the target frame is -1, we can't match the !where yet, but we should prepare to step into it.
                active_where_expects[scope.where].pending_wheres.append(where)
                return
            if where.at_frame_idx is not None:
                # !and {at_frame_idx} is a special case: it cannot contain !where nodes, so there's no point checking it
                # when the parent !where is not in the current frame (frame_idx=0), and we match its other conditions
                # against the requested frame index.
                assert where.is_and, "illegal `at_frame_idx` property for !where"
                if target_frame_idx != 0 or where.at_frame_idx >= len(step_info.frames):
                    return
                target_frame_idx = where.at_frame_idx
            labels = script.get_labels(
                expected_file or step_info.frames[target_frame_idx].loc.path
            )
            match_result = match_where_to_frame(
                where,
                target_frame_idx,
                step_info,
                labels,
                match_context,
            )
            if match_result == WhereFrameMatchResult.TRUE:
                active_where_expects[where] = WhereMatchResult(target_frame_idx)
            elif match_result == WhereFrameMatchResult.EARLY:
                early_wheres.add(where)
            return
        # For this !where, search for the rootmost stack frame that matches it.
        for frame_idx, frame in reversed(list(enumerate(step_info.frames))):
            labels = script.get_labels(expected_file or frame.loc.path)
            match_result = match_where_to_frame(
                where,
                frame_idx,
                step_info,
                labels,
                match_context,
                script.root_scope.file,
            )
            if match_result == WhereFrameMatchResult.TRUE:
                active_where_expects[where] = WhereMatchResult(frame_idx)
                return
            if match_result == WhereFrameMatchResult.EARLY:
                early_wheres.add(where)
                return


    # As we visit the script nodes in pre-order traversal, we can always assume that an expect's parent !where
    # has already been visited, and thus should have an entry in active_where_expects if it is active.
    def get_active_expects(expect: Expect, expected_value, scope: Scope):
        # Active if the matching frame index is either 0, or equal to scope.get_desired_frame_idx() if it is not None.
        if scope.where in active_where_expects and active_where_expects[
            scope.where
        ].frame_idx == (scope.get_desired_frame_idx() or 0):
            active_where_expects[scope.where].active_expects.append(expect)

    def get_active_thens(then: Then, scope: Scope):
        # Active if the matching frame index is either 0, or equal to scope.get_desired_frame_idx() if it is not None.
        if scope.where in active_where_expects and active_where_expects[
            scope.where
        ].frame_idx == (scope.get_desired_frame_idx() or 0):
            active_where_expects[scope.where].active_thens.append(then)

    script.visit_script(
        visit_where=get_active_wheres,
        visit_expect=get_active_expects,
        visit_then=get_active_thens,
    )

    result = StateMatchResult(active_where_expects, early_wheres)
    match_context.update(result)
    return result
