# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Conditional Controller Class for DExTer.-"""


import os
import time
from collections import defaultdict
from itertools import chain

from dex.debugger.DebuggerControllers.ControllerHelpers import in_source_file, update_step_watches
from dex.debugger.DebuggerControllers.DebuggerControllerBase import DebuggerControllerBase
from dex.debugger.DebuggerBase import DebuggerBase
from dex.utils.Exceptions import DebuggerException


class ConditionalBpRange:
    """Represents a conditional range of breakpoints within a source file descending from
    one line to another."""

    def __init__(self, expression: str, path: str, range_from: int, range_to: int, values: list):
        self.expression = expression
        self.path = path
        self.range_from = range_from
        self.range_to = range_to
        self.conditional_values = values

    def get_conditional_expression_list(self):
        conditional_list = []
        for value in self.conditional_values:
            # (<expression>) == (<value>)
            conditional_expression = '({}) == ({})'.format(self.expression, value)
            conditional_list.append(conditional_expression)
        return conditional_list


class ConditionalController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
      self.context = context
      self.step_collection = step_collection
      self._conditional_bps = None
      self._watches = set()
      self._step_index = 0
      self._build_conditional_bps()
      self._path_and_line_to_conditional_bp = defaultdict(list)
      self._pause_between_steps = context.options.pause_between_steps
      self._max_steps = context.options.max_steps

    def _build_conditional_bps(self):
        commands = self.step_collection.commands
        self._conditional_bps = []
        try:
            limit_commands = commands['DexLimitSteps']
            for lc in limit_commands:
                conditional_bp = ConditionalBpRange(
                  lc.expression,
                  lc.path,
                  lc.from_line,
                  lc.to_line,
                  lc.values)
                self._conditional_bps.append(conditional_bp)
        except KeyError:
            raise DebuggerException('Missing DexLimitSteps commands, cannot conditionally step.')

    def _set_conditional_bps(self):
        # When we break in the debugger we need a quick and easy way to look up
        # which conditional bp we've breaked on.
        for cbp in self._conditional_bps:
            conditional_bp_list = self._path_and_line_to_conditional_bp[(cbp.path, cbp.range_from)]
            conditional_bp_list.append(cbp)

        # Set break points only on the first line of any conditional range, we'll set
        # more break points for a range when the condition is satisfied.
        for cbp in self._conditional_bps:
            for cond_expr in cbp.get_conditional_expression_list():
                self.debugger.add_conditional_breakpoint(cbp.path, cbp.range_from, cond_expr)

    def _conditional_met(self, cbp):
        for cond_expr in cbp.get_conditional_expression_list():
            valueIR = self.debugger.evaluate_expression(cond_expr)
            if valueIR.type_name == 'bool' and valueIR.value == 'true':
                return True
        return False

    def _run_debugger_custom(self):
        # TODO: Add conditional and unconditional breakpoint support to dbgeng.
        if self.debugger.get_name() == 'dbgeng':
            raise DebuggerException('DexLimitSteps commands are not supported by dbgeng')

        self.step_collection.clear_steps()
        self._set_conditional_bps()

        for command_obj in chain.from_iterable(self.step_collection.commands.values()):
            self._watches.update(command_obj.get_watches())

        self.debugger.launch()
        time.sleep(self._pause_between_steps) 
        while not self.debugger.is_finished:
            while self.debugger.is_running:
                pass

            step_info = self.debugger.get_step_info(self._watches, self._step_index)
            if step_info.current_frame:
                self._step_index += 1
                update_step_watches(step_info, self._watches, self.step_collection.commands)
                self.step_collection.new_step(self.context, step_info)

                loc = step_info.current_location
                conditional_bp_key = (loc.path, loc.lineno)
                if conditional_bp_key in self._path_and_line_to_conditional_bp:

                    conditional_bps = self._path_and_line_to_conditional_bp[conditional_bp_key]
                    for cbp in conditional_bps:
                        if self._conditional_met(cbp):
                            # Unconditional range should ignore first line as that's the
                            # conditional bp we just hit and should be inclusive of final line
                            for line in range(cbp.range_from + 1, cbp.range_to + 1):
                                self.debugger.add_conditional_breakpoint(cbp.path, line, condition='')

            # Clear any uncondtional break points at this loc.
            self.debugger.delete_conditional_breakpoint(file_=loc.path, line=loc.lineno, condition='')
            self.debugger.go()
            time.sleep(self._pause_between_steps)
