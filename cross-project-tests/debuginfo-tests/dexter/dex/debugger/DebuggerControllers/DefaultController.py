# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Default class for controlling debuggers."""

from itertools import chain
import os
import time

from dex.debugger.DebuggerControllers.DebuggerControllerBase import DebuggerControllerBase
from dex.debugger.DebuggerControllers.ControllerHelpers import in_source_file, update_step_watches
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException

class DefaultController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
        self.context = context
        self.step_collection = step_collection
        self.source_files = self.context.options.source_files
        self.watches = set()
        self.step_index = 0

    def _break_point_all_lines(self):
        for s in self.context.options.source_files:
            with open(s, 'r') as fp:
                num_lines = len(fp.readlines())
            for line in range(1, num_lines + 1):
                try:
                   self.debugger.add_breakpoint(s, line)
                except DebuggerException:
                   raise LoadDebuggerException(DebuggerException.msg)

    def _run_debugger_custom(self):
        self.step_collection.debugger = self.debugger.debugger_info
        self._break_point_all_lines()
        self.debugger.launch()

        for command_obj in chain.from_iterable(self.step_collection.commands.values()):
            self.watches.update(command_obj.get_watches())

        max_steps = self.context.options.max_steps
        for _ in range(max_steps):
            while self.debugger.is_running:
                pass

            if self.debugger.is_finished:
                break

            self.step_index += 1
            step_info = self.debugger.get_step_info(self.watches, self.step_index)

            if step_info.current_frame:
                update_step_watches(step_info, self.watches, self.step_collection.commands)
                self.step_collection.new_step(self.context, step_info)

            if in_source_file(self.source_files, step_info):
                self.debugger.step()
            else:
                self.debugger.go()

            time.sleep(self.context.options.pause_between_steps)
        else:
            raise DebuggerException(
                'maximum number of steps reached ({})'.format(max_steps))
