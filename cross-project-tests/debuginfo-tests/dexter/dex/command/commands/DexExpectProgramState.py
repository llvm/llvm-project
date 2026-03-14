# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Command for specifying a partial or complete state for the program to enter
during execution.
"""

from itertools import chain

from dex.command.CommandBase import CommandBase, StepExpectInfo
from dex.dextIR import ProgramState, SourceLocation, StackFrame, DextIR


def frame_from_dict(source: dict) -> StackFrame:
    if "location" in source:
        assert isinstance(source["location"], dict)
        source["location"] = SourceLocation(**source["location"])
    return StackFrame(**source)


def state_from_dict(source: dict) -> ProgramState:
    if "frames" in source:
        assert isinstance(source["frames"], list)
        source["frames"] = list(map(frame_from_dict, source["frames"]))
    return ProgramState(**source)


class DexExpectProgramState(CommandBase):
    """Expect to see a given program `state` a certain numer of `times`.

    DexExpectProgramState(state [,**times])

    See Commands.md for more info.
    """

    def __init__(self, *args, **kwargs):
        if len(args) != 1:
            raise TypeError("expected exactly one unnamed arg")

        self.program_state_text = str(args[0])

        self.expected_program_state = state_from_dict(args[0])

        self.times = kwargs.pop("times", -1)
        if kwargs:
            raise TypeError("unexpected named args: {}".format(", ".join(kwargs)))

        # Step indices at which the expected program state was encountered.
        self.encounters = []

        super(DexExpectProgramState, self).__init__()

    @staticmethod
    def get_name():
        return __class__.__name__

    def get_watches(self):
        frame_expects = set()
        for idx, frame in enumerate(self.expected_program_state.frames):
            path = (
                frame.location.path
                if frame.location and frame.location.path
                else self.path
            )
            line_range = (
                range(frame.location.lineno, frame.location.lineno + 1)
                if frame.location and frame.location.lineno
                else None
            )
            for watch in frame.watches:
                frame_expects.add(
                    StepExpectInfo(
                        expression=watch,
                        path=path,
                        frame_idx=idx,
                        line_range=line_range,
                    )
                )
        return frame_expects

    def eval(self, step_collection: DextIR) -> bool:
        for step in step_collection.steps:
            if self.expected_program_state.match(step.program_state):
                self.encounters.append(step.step_index)

        return (
            self.times < 0 < len(self.encounters) or len(self.encounters) == self.times
        )
