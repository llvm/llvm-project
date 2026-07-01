# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger state, such as the call stack, conditions, or historical state (e.g. breakpoint
hitcounts) to descriptions of expected state in a DexterScript."""

from collections import Counter
from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Set

from dex.dextIR import FrameIR, StepIR
from dex.test_script import DexterScript, Scope
from dex.test_script.Nodes import Expect, FileLabels, Where, Then


class StateMatchContext:
    """Class that holds any state needed for matching state nodes to debugger state across a run."""

    def __init__(self):
        self.where_hit_counts: Counter[Where] = Counter()
        self.expired_wheres: Set[Where] = set()
        self._last_match_result: Optional[StateMatchResult] = None

    def where_hit_is_new(self, where: Where, step: StepIR) -> bool:
        """Returns True if the current step can be counted as a new "hit" for `where`, assuming that `where` was hit in
        this step (but does not check if `where` has actually been hit in the current step).
        """
        # If this is the first step, all hits are new.
        if self._last_match_result is None:
            return True
        # If this !where did not appear in any frame in the previous step, this is a fresh hit.
        if where not in self._last_match_result:
            return True
        # If !where uses a function breakpoint and that breakpoint was hit this step, this is a fresh hit.
        if where.function and not where.lines and where in step.hit_where_bps:
            return True
        return False

    def add_hit_if_where_hit_is_new(self, where: Where, step: StepIR) -> bool:
        """Checks whether the current step can be counted as a new "hit" for `where`. Increments `where`'s hit count if
        it has a new hit, and returns True iff so."""
        assert (
            where.for_hit_count is not None
        ), "Tried to add hit count for !where without for_hit_count?"
        if self.where_hit_is_new(where, step):
            self.where_hit_counts[where] += 1
            if self.where_hit_counts[where] >= where.for_hit_count:
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


def _match_where_to_frame(
    where: Where,
    frame: FrameIR,
    labels: FileLabels,
    context: StateMatchContext,
    default_path: Optional[str] = None,
) -> bool:
    """A very simple matcher, returns True iff `where` matches `frame`."""
    file = where.file
    if not file and where.lines and not where.function:
        file = default_path
    if file is not None and not is_subpath(file, frame.loc.path):
        return False
    if where.function is not None:
        fn = frame.function
        if "(" in fn:
            fn = fn.split("(")[0]
        if where.function != fn:
            return False
    if where.lines is not None:
        if frame.loc.lineno not in where.get_lines(labels):
            return False
    if where.for_hit_count is not None:
        where_hit_count = context.where_hit_counts[where]
        if where_hit_count > where.for_hit_count:
            return False
    if where.after_hit_count is not None or where.conditions is not None:
        raise NotImplementedError(
            "!where hit counts and conditions currently unsupported."
        )
    return True


def match_where_to_frame(
    where: Where,
    frame: FrameIR,
    step: StepIR,
    labels: FileLabels,
    context: StateMatchContext,
    default_path: Optional[str] = None,
) -> bool:
    """Returns True if `where` matches `frame`. As part of this check, we perform the check once, and if necessary we
    may increment `where`'s hit count and check again."""
    result = _match_where_to_frame(where, frame, labels, context, default_path)
    if result == True and where.for_hit_count is not None:
        if context.add_hit_if_where_hit_is_new(where, step):
            result = _match_where_to_frame(where, frame, labels, context, default_path)
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


StateMatchResult = Dict[Where, WhereMatchResult]


def get_active_where_matches(
    script: DexterScript, step_info: StepIR, match_context: StateMatchContext
) -> Dict[Where, WhereMatchResult]:
    """Match the script against the step_info, producing a dict that maps each !where that matches a stack frame to the
    index of the (rootmost) stack frame that it matches, and if the frame that it matches is the current stack frame
    (i.e. the frame index is 0), also includes a list of every direct child !expect node for that !where.
    """
    active_where_expects: Dict[Where, WhereMatchResult] = {}

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
            if match_where_to_frame(
                where,
                step_info.frames[target_frame_idx],
                step_info,
                labels,
                match_context,
            ):
                active_where_expects[where] = WhereMatchResult(target_frame_idx)
            return
        # For this !where, search for the rootmost stack frame that matches it.
        matching_frame_idx = None
        for frame_idx, frame in reversed(list(enumerate(step_info.frames))):
            labels = script.get_labels(expected_file or frame.loc.path)
            if match_where_to_frame(
                where, frame, step_info, labels, match_context, script.root_scope.file
            ):
                matching_frame_idx = frame_idx
                break

        if matching_frame_idx is not None:
            active_where_expects[where] = WhereMatchResult(matching_frame_idx)

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

    match_context.update(active_where_expects)
    return active_where_expects
