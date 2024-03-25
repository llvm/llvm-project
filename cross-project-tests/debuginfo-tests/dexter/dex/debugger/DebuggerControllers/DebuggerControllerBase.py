# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Abstract Base class for controlling debuggers."""

import abc


class DebuggerControllerBase(object, metaclass=abc.ABCMeta):
    def __init__(self, context, step_collection):
        self.context = context
        self.step_collection = step_collection

    @abc.abstractclassmethod
    def _run_debugger_custom(self):
        """Specify your own implementation of run_debugger_custom in your own
        controller.
        """
        pass

    def run_debugger(self, debugger):
        """Responsible for correctly launching and tearing down the debugger."""
        self.debugger = debugger

        # Fetch command line options, if any.
        the_cmdline = []
        commands = self.step_collection.commands
        if "DexCommandLine" in commands:
            cmd_line_objs = commands["DexCommandLine"]
            assert len(cmd_line_objs) == 1
            cmd_line_obj = cmd_line_objs[0]
            the_cmdline = cmd_line_obj.the_cmdline

        with self.debugger:
            if not self.debugger.loading_error:
                self._run_debugger_custom(the_cmdline)

        # We may need to pickle this debugger controller after running the
        # debugger. Debuggers are not picklable objects, so set to None.
        self.debugger = None
