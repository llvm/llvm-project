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


class BreakpointRange:
    """A range of breakpoints and a set of conditions.

    The leading breakpoint (on line `range_from`) is always active.

    When the leading breakpoint is hit the trailing range should be activated
    when `expression` evaluates to any value in `values`. If there are no
    conditions (`expression` is None) then the trailing breakpoint range should
    always be activated upon hitting the leading breakpoint.

    Args:
       expression: None for no conditions, or a str expression to compare
       against `values`.

       hit_count: None for no limit, or int to set the number of times the
                  leading breakpoint is triggered before it is removed.
    """

    def __init__(self, expression: str, path: str, range_from: int, range_to: int,
                 values: list, hit_count: int, finish_on_remove: bool):
        self.expression = expression
        self.path = path
        self.range_from = range_from
        self.range_to = range_to
        self.conditional_values = values
        self.max_hit_count = hit_count
        self.current_hit_count = 0
        self.finish_on_remove = finish_on_remove

    def has_conditions(self):
        return self.expression != None

    def get_conditional_expression_list(self):
        conditional_list = []
        for value in self.conditional_values:
            # (<expression>) == (<value>)
            conditional_expression = '({}) == ({})'.format(self.expression, value)
            conditional_list.append(conditional_expression)
        return conditional_list

    def add_hit(self):
        self.current_hit_count += 1

    def should_be_removed(self):
        if self.max_hit_count == None:
            return False
        return self.current_hit_count >= self.max_hit_count


class ConditionalController(DebuggerControllerBase):
    def __init__(self, context, step_collection):
      self._bp_ranges = None
      self._watches = set()
      self._step_index = 0
      self._pause_between_steps = context.options.pause_between_steps
      self._max_steps = context.options.max_steps
      # Map {id: BreakpointRange}
      self._leading_bp_handles = {}
      super(ConditionalController, self).__init__(context, step_collection)
      self._build_bp_ranges()

    def _build_bp_ranges(self):
        commands = self.step_collection.commands
        self._bp_ranges = []
        try:
            limit_commands = commands['DexLimitSteps']
            for lc in limit_commands:
                bpr = BreakpointRange(
                  lc.expression,
                  lc.path,
                  lc.from_line,
                  lc.to_line,
                  lc.values,
                  lc.hit_count,
                  False)
                self._bp_ranges.append(bpr)
        except KeyError:
            raise DebuggerException('Missing DexLimitSteps commands, cannot conditionally step.')
        if 'DexFinishTest' in commands:
            finish_commands = commands['DexFinishTest']
            for ic in finish_commands:
                bpr = BreakpointRange(
                  ic.expression,
                  ic.path,
                  ic.on_line,
                  ic.on_line,
                  ic.values,
                  ic.hit_count + 1,
                  True)
                self._bp_ranges.append(bpr)

    def _set_leading_bps(self):
        # Set a leading breakpoint for each BreakpointRange, building a
        # map of {leading bp id: BreakpointRange}.
        for bpr in self._bp_ranges:
            if bpr.has_conditions():
                # Add a conditional breakpoint for each condition.
                for cond_expr in bpr.get_conditional_expression_list():
                    id = self.debugger.add_conditional_breakpoint(bpr.path,
                                                                  bpr.range_from,
                                                                  cond_expr)
                    self._leading_bp_handles[id] = bpr
            else:
                # Add an unconditional breakpoint.
                id = self.debugger.add_breakpoint(bpr.path, bpr.range_from)
                self._leading_bp_handles[id] = bpr

    def _run_debugger_custom(self, cmdline):
        # TODO: Add conditional and unconditional breakpoint support to dbgeng.
        if self.debugger.get_name() == 'dbgeng':
            raise DebuggerException('DexLimitSteps commands are not supported by dbgeng')

        self.step_collection.clear_steps()
        self._set_leading_bps()

        for command_obj in chain.from_iterable(self.step_collection.commands.values()):
            self._watches.update(command_obj.get_watches())

        self.debugger.launch(cmdline)
        time.sleep(self._pause_between_steps)

        exit_desired = False

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
                    # See if this is one of our leading breakpoints.
                    bpr = self._leading_bp_handles[bp_id]
                except KeyError:
                    # This is a trailing bp. Mark it for removal.
                    bp_to_delete.append(bp_id)
                    continue

                bpr.add_hit()
                if bpr.should_be_removed():
                    if bpr.finish_on_remove:
                        exit_desired = True
                    bp_to_delete.append(bp_id)
                    del self._leading_bp_handles[bp_id]
                # Add a range of trailing breakpoints covering the lines
                # requested in the DexLimitSteps command. Ignore first line as
                # that's covered by the leading bp we just hit and include the
                # final line.
                for line in range(bpr.range_from + 1, bpr.range_to + 1):
                    self.debugger.add_breakpoint(bpr.path, line)

            # Remove any trailing or expired leading breakpoints we just hit.
            self.debugger.delete_breakpoints(bp_to_delete)

            if exit_desired:
                break
            self.debugger.go()
            time.sleep(self._pause_between_steps)
