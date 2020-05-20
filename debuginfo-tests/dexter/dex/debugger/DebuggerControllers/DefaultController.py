# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base class for controlling debuggers."""

from itertools import chain
import os
import time

from dex.debugger.DebuggerControllers.DebuggerControllerBase import DebuggerControllerBase
from dex.utils.Exceptions import DebuggerException

class DefaultController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
        self.context = context
        self.step_collection = step_collection
        self.watches = set()
        self.step_index = 0

    def _update_step_watches(self, step_info):
        watch_cmds = ['DexUnreachable', 'DexExpectStepOrder']
        towatch = chain.from_iterable(self.step_collection.commands[x]
                                      for x in watch_cmds
                                      if x in self.step_collection.commands)
        try:
            # Iterate over all watches of the types named in watch_cmds
            for watch in towatch:
                loc = step_info.current_location
                if (os.path.exists(loc.path)
                        and os.path.samefile(watch.path, loc.path)
                        and watch.lineno == loc.lineno):
                    result = watch.eval(step_info)
                    step_info.watches.update(result)
                    break
        except KeyError:
            pass

    def _break_point_all_lines(self):
        for s in self.context.options.source_files:
            with open(s, 'r') as fp:
                num_lines = len(fp.readlines())
            for line in range(1, num_lines + 1):
                self.debugger.add_breakpoint(s, line)

    def _in_source_file(self, step_info):
        if not step_info.current_frame:
            return False
        if not step_info.current_location.path:
            return False
        if not os.path.exists(step_info.current_location.path):
            return False
        return any(os.path.samefile(step_info.current_location.path, f) \
                   for f in self.context.options.source_files)

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
                self._update_step_watches(step_info)
                self.step_collection.new_step(self.context, step_info)

            if self._in_source_file(step_info):
                self.debugger.step()
            else:
                self.debugger.go()

            time.sleep(self.context.options.pause_between_steps)
        else:
            raise DebuggerException(
                'maximum number of steps reached ({})'.format(max_steps))
