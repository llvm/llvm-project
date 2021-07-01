# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Abstract Base class for controlling debuggers."""

import abc

class DebuggerControllerBase(object, metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def _run_debugger_custom(self):
        """Specify your own implementation of run_debugger_custom in your own
        controller.
        """
        pass

    def run_debugger(self, debugger):
        """Responsible for correctly launching and tearing down the debugger.
        """
        self.debugger = debugger
        with self.debugger:
            self._run_debugger_custom()
        # We may need to pickle this debugger controller after running the
        # debugger. Debuggers are not picklable objects, so set to None.
        self.debugger = None
