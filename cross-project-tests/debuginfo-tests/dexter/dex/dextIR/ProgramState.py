# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Set of data classes for representing the complete debug program state at a
fixed point in execution.
"""

import os

from collections import OrderedDict
from typing import List

class SourceLocation:
    def __init__(self, path: str = None, lineno: int = None, column: int = None):
        if path:
            path = os.path.normcase(path)
        self.path = path
        self.lineno = lineno
        self.column = column

    def __str__(self):
        return '{}({}:{})'.format(self.path, self.lineno, self.column)

    def match(self, other) -> bool:
        """Returns true iff all the properties that appear in `self` have the
        same value in `other`, but not necessarily vice versa.
        """
        if not other or not isinstance(other, SourceLocation):
            return False

        if self.path and (self.path != other.path):
            return False

        if self.lineno and (self.lineno != other.lineno):
            return False

        if self.column and (self.column != other.column):
            return False

        return True


class StackFrame:
    def __init__(self,
                 function: str = None,
                 is_inlined: bool = None,
                 location: SourceLocation = None,
                 watches: OrderedDict = None):
        if watches is None:
            watches = {}

        self.function = function
        self.is_inlined = is_inlined
        self.location = location
        self.watches = watches

    def __str__(self):
        return '{}{}: {} | {}'.format(
            self.function,
            ' (inlined)' if self.is_inlined else '',
            self.location,
            {k: str(self.watches[k]) for k in self.watches})

    def match(self, other) -> bool:
        """Returns true iff all the properties that appear in `self` have the
        same value in `other`, but not necessarily vice versa.
        """
        if not other or not isinstance(other, StackFrame):
            return False

        if self.location and not self.location.match(other.location):
            return False

        if self.watches:
            for name in iter(self.watches):
                try:
                    if isinstance(self.watches[name], dict):
                        for attr in iter(self.watches[name]):
                            if (getattr(other.watches[name], attr, None) !=
                                    self.watches[name][attr]):
                                return False
                    else:
                        if other.watches[name].value != self.watches[name]:
                            return False
                except KeyError:
                    return False

        return True

class ProgramState:
    def __init__(self, frames: List[StackFrame] = None):
        self.frames = frames

    def __str__(self):
        return '\n'.join(map(
            lambda enum: 'Frame {}: {}'.format(enum[0], enum[1]),
            enumerate(self.frames)))

    def match(self, other) -> bool:
        """Returns true iff all the properties that appear in `self` have the
        same value in `other`, but not necessarily vice versa.
        """
        if not other or not isinstance(other, ProgramState):
            return False

        if self.frames:
            for idx, frame in enumerate(self.frames):
                try:
                    if not frame.match(other.frames[idx]):
                        return False
                except (IndexError, KeyError):
                    return False

        return True
