# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utilities for matching debugger state, such as the call stack, conditions, or historical state (e.g. breakpoint
hitcounts) to descriptions of expected state in a DexterScript."""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Tuple

from dex.dextIR import FrameIR, StepIR
from dex.test_script import DexterScript, Scope
from dex.test_script.Nodes import Expect, Value, Where


def is_subpath(subpath: str, superpath: str) -> bool:
    """Returns True if subpath is a trailing subpath of superpath, i.e. if `superpath` ends with `subpath` after
    normalizing both paths."""
    normalized_subpath: str = os.path.normcase(os.path.normpath(subpath))
    normalized_superpath: str = os.path.normcase(os.path.normpath(superpath))
    return normalized_superpath.endswith(normalized_subpath)


# A very simple matcher, returns True iff `where` matches `frame`.
def match_where_to_frame(
    where: Where,
    frame: FrameIR,
) -> bool:
    """A very simple matcher, returns True iff `where` matches `frame`."""
    if where.file is not None and not is_subpath(where.file, frame.loc.path):
        return False
    if where.function is not None:
        fn = frame.function
        if "(" in fn:
            fn = fn.split("(")[0]
        if where.function != fn:
            return False
    if where.lines is not None:
        if frame.loc.lineno not in where.get_lines():
            return False
    if (
        where.for_hit_count is not None
        or where.after_hit_count is not None
        or where.conditions is not None
    ):
        raise NotImplementedError(
            "!where hit counts and conditions currently unsupported."
        )
    return True


@dataclass
class WhereMatchResult:
    """Class storing the result of a single !where matched against a stack frame. The primary information stored is just
    the frame index that the !where matched against; for convenience, the children of the !where are also included.
    """

    frame_idx: int
    active_expects: List[Value] = field(default_factory=list)
    pending_wheres: List[Where] = field(default_factory=list)


def get_active_where_matches(
    script: DexterScript, step_info: StepIR
) -> Dict[Where, WhereMatchResult]:
    """Match the script against the step_info, producing a dict that maps each !where that matches a stack frame to the
    index of the (rootmost) stack frame that it matches, and if the frame that it matches is the current stack frame
    (i.e. the frame index is 0), also includes a list of every direct child !expect node for that !where.
    """
    active_where_expects: Dict[Where, WhereMatchResult] = {}

    def get_active_wheres(where: Where, scope: Scope):
        # For nested !wheres, we must match a specific frame relative to the parent !where.
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
            if match_where_to_frame(where, step_info.frames[target_frame_idx]):
                active_where_expects[where] = WhereMatchResult(target_frame_idx)
            return
        # For this !where, search for the rootmost stack frame that matches it.
        matching_frame_idx = next(
            (
                frame_idx
                for frame_idx, frame in reversed(list(enumerate(step_info.frames)))
                if match_where_to_frame(where, frame)
            ),
            None,
        )
        if matching_frame_idx is not None:
            active_where_expects[where] = WhereMatchResult(matching_frame_idx)

    # As we visit the script nodes in pre-order traversal, we can always assume that an expect's parent !where
    # has already been visited, and thus should have an entry in active_where_expects if it is active.
    def get_active_expects(expect: Expect, expected_value, scope: Scope):
        assert isinstance(
            expect, Value
        ), "Values should be the only type of expect possible!"
        if (
            scope.where in active_where_expects
            and active_where_expects[scope.where].frame_idx == 0
        ):
            active_where_expects[scope.where].active_expects.append(expect)

    script.visit_script(visit_where=get_active_wheres, visit_expect=get_active_expects)

    return active_where_expects
