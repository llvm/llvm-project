# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from collections import OrderedDict
import os
from typing import List

from dex.dextIR.BuilderIR import BuilderIR
from dex.dextIR.DebuggerIR import DebuggerIR
from dex.dextIR.StepIR import StepIR, StepKind


def _step_kind_func(context, step):
    if (step.current_location.path is None or
        not os.path.exists(step.current_location.path)):
        return StepKind.FUNC_UNKNOWN

    if any(os.path.samefile(step.current_location.path, f)
           for f in context.options.source_files):
        return StepKind.FUNC

    return StepKind.FUNC_EXTERNAL


class DextIR:
    """A full Dexter test report.

    This is composed of all the other *IR classes. They are used together to
    record Dexter inputs and the resultant debugger steps, providing a single
    high level access container.

    The Heuristic class works with dexter commands and the generated DextIR to
    determine the debugging score for a given test.

    Args:
        commands: { name (str), commands (list[CommandIR])
    """

    def __init__(self,
                 dexter_version: str,
                 executable_path: str,
                 source_paths: List[str],
                 builder: BuilderIR = None,
                 debugger: DebuggerIR = None,
                 commands: OrderedDict = None):
        self.dexter_version = dexter_version
        self.executable_path = executable_path
        self.source_paths = source_paths
        self.builder = builder
        self.debugger = debugger
        self.commands = commands
        self.steps: List[StepIR] = []

    def __str__(self):
        colors = 'rgby'
        st = '## BEGIN ##\n'
        color_idx = 0
        for step in self.steps:
            if step.step_kind in (StepKind.FUNC, StepKind.FUNC_EXTERNAL,
                                  StepKind.FUNC_UNKNOWN):
                color_idx += 1

            color = colors[color_idx % len(colors)]
            st += '<{}>{}</>\n'.format(color, step)
        st += '## END ({} step{}) ##\n'.format(
            self.num_steps, '' if self.num_steps == 1 else 's')
        return st

    @property
    def num_steps(self):
        return len(self.steps)

    def _get_prev_step_in_this_frame(self, step):
        """Find the most recent step in the same frame as `step`.

        Returns:
            StepIR or None if there is no previous step in this frame.
        """
        return next((s for s in reversed(self.steps)
            if s.current_function == step.current_function
            and s.num_frames == step.num_frames), None)

    def _get_new_step_kind(self, context, step):
        if step.current_function is None:
            return StepKind.UNKNOWN

        if len(self.steps) == 0:
            return _step_kind_func(context, step)

        prev_step = self.steps[-1]

        if prev_step.current_function is None:
            return StepKind.UNKNOWN

        if prev_step.num_frames < step.num_frames:
            return _step_kind_func(context, step)

        if prev_step.num_frames > step.num_frames:
            frame_step = self._get_prev_step_in_this_frame(step)
            prev_step = frame_step if frame_step is not None else prev_step

        # We're in the same func as prev step, check lineo.
        if prev_step.current_location.lineno > step.current_location.lineno:
            return StepKind.VERTICAL_BACKWARD

        if prev_step.current_location.lineno < step.current_location.lineno:
            return StepKind.VERTICAL_FORWARD

        # We're on the same line as prev step, check column.
        if prev_step.current_location.column > step.current_location.column:
            return StepKind.HORIZONTAL_BACKWARD

        if prev_step.current_location.column < step.current_location.column:
            return StepKind.HORIZONTAL_FORWARD

        # This step is in exactly the same location as the prev step.
        return StepKind.SAME

    def new_step(self, context, step):
        assert isinstance(step, StepIR), type(step)
        step.step_kind = self._get_new_step_kind(context, step)
        self.steps.append(step)
        return step

    def clear_steps(self):
        self.steps.clear()
