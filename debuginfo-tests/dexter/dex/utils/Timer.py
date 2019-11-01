# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""RAII-style timer class to be used with a 'with' statement to get wall clock
time for the contained code.
"""

import sys
import time


def _indent(indent):
    return '| ' * indent


class Timer(object):
    fn = sys.stdout.write
    display = False
    indent = 0

    def __init__(self, name=None):
        self.name = name
        self.start = self.now

    def __enter__(self):
        Timer.indent += 1
        if Timer.display and self.name:
            indent = _indent(Timer.indent - 1) + ' _'
            Timer.fn('{}\n'.format(_indent(Timer.indent - 1)))
            Timer.fn('{} start {}\n'.format(indent, self.name))
        return self

    def __exit__(self, *args):
        if Timer.display and self.name:
            indent = _indent(Timer.indent - 1) + '|_'
            Timer.fn('{} {} time taken: {:0.1f}s\n'.format(
                indent, self.name, self.elapsed))
            Timer.fn('{}\n'.format(_indent(Timer.indent - 1)))
        Timer.indent -= 1

    @property
    def elapsed(self):
        return self.now - self.start

    @property
    def now(self):
        return time.time()
