# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os


class LocIR:
    """Data class which represents a source location."""

    def __init__(self, path: str, lineno: int, column: int):
        if path:
            path = os.path.normcase(path)
        self.path = path
        self.lineno = lineno
        self.column = column

    def __str__(self):
        return '{}({}:{})'.format(self.path, self.lineno, self.column)

    def __eq__(self, rhs):
        return (os.path.exists(self.path) and os.path.exists(rhs.path)
                and os.path.samefile(self.path, rhs.path)
                and self.lineno == rhs.lineno
                and self.column == rhs.column)

    def __lt__(self, rhs):
        if self.path != rhs.path:
            return False

        if self.lineno == rhs.lineno:
            return self.column < rhs.column

        return self.lineno < rhs.lineno

    def __gt__(self, rhs):
        if self.path != rhs.path:
            return False

        if self.lineno == rhs.lineno:
            return self.column > rhs.column

        return self.lineno > rhs.lineno
