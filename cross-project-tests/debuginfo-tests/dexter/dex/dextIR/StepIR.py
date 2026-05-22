# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Classes which are used to represent debugger steps."""

import json

from collections import OrderedDict
from typing import List
from enum import Enum
from dex.dextIR.FrameIR import FrameIR
from dex.dextIR.LocIR import LocIR
from dex.dextIR.ProgramState import ProgramState


class StopReason(Enum):
    BREAKPOINT = 0
    STEP = 1
    PROGRAM_EXIT = 2
    ERROR = 3
    OTHER = 4


class StepKind(Enum):
    FUNC = 0
    FUNC_EXTERNAL = 1
    FUNC_UNKNOWN = 2
    VERTICAL_FORWARD = 3
    SAME = 4
    VERTICAL_BACKWARD = 5
    UNKNOWN = 6
    HORIZONTAL_FORWARD = 7
    HORIZONTAL_BACKWARD = 8


class StepIR:
    """A debugger step.

    Args:
        watches (OrderedDict): { expression (str), result (ValueIR) }
    """

    def __init__(
        self,
        step_index: int,
        stop_reason: StopReason,
        frames: List[FrameIR],
        step_kind: StepKind = None,
        watches: OrderedDict = None,
        program_state: ProgramState = None,
    ):
        self.step_index = step_index
        self.step_kind = step_kind
        self.stop_reason = stop_reason
        self.program_state = program_state

        if frames is None:
            frames = []
        self.frames = frames

        if watches is None:
            watches = {}
        self.watches = watches
        self.hit_fn_bps: List[str] = []

    def __str__(self):
        try:
            frame = self.current_frame
            frame_info = (
                frame.function,
                frame.loc.path,
                frame.loc.lineno,
                frame.loc.column,
            )
        except AttributeError:
            frame_info = (None, None, None, None)

        step_info = (
            (self.step_index,)
            + frame_info
            + (str(self.stop_reason), str(self.step_kind), [w for w in self.watches])
        )

        return "{}{}".format(".   " * (self.num_frames - 1), json.dumps(step_info))

    @property
    def num_frames(self):
        return len(self.frames)

    @property
    def current_frame(self):
        if not len(self.frames):
            return None
        return self.frames[0]

    @property
    def current_function(self):
        try:
            return self.current_frame.function
        except AttributeError:
            return None

    @property
    def current_location(self):
        try:
            return self.current_frame.loc
        except AttributeError:
            return LocIR(path=None, lineno=None, column=None)

    def detailed_print(self) -> List[str]:
        lines: List[str] = []
        lines.append(f"Step {self.step_index}")
        lines.append(f"Stopped for {self.stop_reason}")
        lines.append(f"Step was {self.step_kind}")
        lines.append(f"Frames:")
        for idx, frame in enumerate(self.frames):
            lines.append(f"  Frame {idx}:")
            fn_text = (
                "[Inline Function] " if frame.is_inlined else ""
            ) + frame.function
            lines.append(f"    {fn_text}")
            lines.append(f"    {frame.loc}")
            lines.append(f"    $pc = {frame.instruction_addr}")

        if self.watches:
            lines.append(f"Variables:")
            for value in sorted(self.watches.values(), key=lambda v: v.expression):
                lines.append(f"  {value}")
        return lines
