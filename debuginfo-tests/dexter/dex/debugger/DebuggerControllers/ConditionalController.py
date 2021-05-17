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
      self._conditional_bp_ranges = None
      self._build_conditional_bp_ranges()
      self._watches = set()
      self._step_index = 0
      self._pause_between_steps = context.options.pause_between_steps
      self._max_steps = context.options.max_steps
      # Map {id: ConditionalBpRange}
      self._conditional_bp_handles = {}

    def _build_conditional_bp_ranges(self):
        commands = self.step_collection.commands
        self._conditional_bp_ranges = []
        try:
            limit_commands = commands['DexLimitSteps']
            for lc in limit_commands:
                conditional_bp = ConditionalBpRange(
                  lc.expression,
                  lc.path,
                  lc.from_line,
                  lc.to_line,
                  lc.values)
                self._conditional_bp_ranges.append(conditional_bp)
        except KeyError:
            raise DebuggerException('Missing DexLimitSteps commands, cannot conditionally step.')

    def _set_conditional_bps(self):
        # Set a conditional breakpoint for each ConditionalBpRange and build a
        # map of {id: ConditionalBpRange}.
        for cbp in self._conditional_bp_ranges:
            for cond_expr in cbp.get_conditional_expression_list():
                id = self.debugger.add_conditional_breakpoint(cbp.path,
                                                              cbp.range_from,
                                                              cond_expr)
                self._conditional_bp_handles[id] = cbp

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

            bp_to_delete = []
            for bp_id in self.debugger.get_triggered_breakpoint_ids():
                try:
                    # See if this is one of our conditional breakpoints.
                    cbp = self._conditional_bp_handles[bp_id]
                except KeyError:
                    # This is an unconditional bp. Mark it for removal.
                    bp_to_delete.append(bp_id)
                    continue
                # We have triggered a breakpoint with a condition. Check that
                # the condition has been met.
                if self._conditional_met(cbp):
                    # Add a range of unconditional breakpoints covering the
                    # lines requested in the DexLimitSteps command. Ignore
                    # first line as that's the conditional bp we just hit and
                    # include the final line.
                    for line in range(cbp.range_from + 1, cbp.range_to + 1):
                        self.debugger.add_breakpoint(cbp.path, line)

            # Remove any unconditional breakpoints we just hit.
            for bp_id in bp_to_delete:
                self.debugger.delete_breakpoint(bp_id)

            self.debugger.go()
            time.sleep(self._pause_between_steps)
