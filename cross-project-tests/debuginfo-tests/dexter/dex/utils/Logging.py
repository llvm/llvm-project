# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Utility functions for producing command line warnings."""

from dex.utils import PrettyOutput


class Logger(object):
    def __init__(self, pretty_output: PrettyOutput):
        self.o = pretty_output
        self.error_color = self.o.red
        self.warning_color = self.o.yellow
        self.note_color = self.o.default
        self.verbosity = 1

    def error(self, msg, enable_prefix=True, flag=None):
        if self.verbosity < 0:
            return
        if enable_prefix:
            msg = f"error: {msg}"
        if flag:
            msg = f"{msg} <y>[{flag}]</>"
        self.error_color("{}\n".format(msg), stream=PrettyOutput.stderr)

    def warning(self, msg, enable_prefix=True, flag=None):
        if self.verbosity < 1:
            return
        if enable_prefix:
            msg = f"warning: {msg}"
        if flag:
            msg = f"{msg} <y>[{flag}]</>"
        self.warning_color("{}\n".format(msg), stream=PrettyOutput.stderr)

    def note(self, msg, enable_prefix=True, flag=None):
        if self.verbosity < 2:
            return
        if enable_prefix:
            msg = f"note: {msg}"
        if flag:
            msg = f"{msg} <y>[{flag}]</>"
        self.note_color("{}\n".format(msg), stream=PrettyOutput.stderr)
