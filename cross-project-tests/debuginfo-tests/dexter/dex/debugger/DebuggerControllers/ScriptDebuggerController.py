# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Debugger Controller Class for DExTer, responsible for driving a debugger session, invoking debugger actions and
recording debugger output."""


from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, List, Optional

from dex.debugger.DebuggerControllers.DebuggerControllerBase import (
    DebuggerControllerBase,
)
from dex.debugger.DebuggerBase import DebuggerBase
from dex.debugger.DAP import DAP
from dex.evaluation.StateMatch import StateMatchContext, get_active_where_matches
from dex.test_script.Nodes import Where
from dex.test_script.Script import DexterScript, Scope
from dex.tools import Context
from dex.utils.Timeout import Timeout
from dex.dextIR import DextIR, StepIR


class DebuggerAction(Enum):
    STEP_OVER = 0
    STEP_OUT = 1
    CONTINUE = 2
    EXIT = 3

class ScriptDebuggerController(DebuggerControllerBase):
    """Uses a Dexter Script to drive a debugger session, using "where" nodes to make breakpoint/stepping decisions, and
    "expect" nodes to evaluate variables."""

    def __init__(self, context: Context, step_collection: DextIR):
        super().__init__(context, step_collection)
        self._step_index = 0
        self._pause_between_steps = context.options.pause_between_steps
        self._max_steps = context.options.max_steps
        assert step_collection.script is not None
        self.script: DexterScript = step_collection.script

        # We may need to pickle this debugger controller after running the
        # debugger. Debuggers are not picklable objects, so this starts as None
        # and will be set back to None after we finish running the debugger.
        self.debugger: DebuggerBase = None  # type: ignore

        # Breakpoint IDs currently set for each !where node.
        self._where_bps: Dict[Where, List[int]] = defaultdict(list)

    def add_where_entry_bp(self, where: Where, default_file: Optional[str] = None):
        """Adds a breakpoint to catch when we enter the given !where node."""
        added_ids: List[int] = []
        if where.function:
            if where.lines:
                raise NotImplementedError(
                    f"Implementation does not currently handle !where with function and lines: {where}"
                )
            added_ids.append(self.debugger.add_function_breakpoint(where.function))
        elif where.lines:
            # We prefer an explicit file, but we make a special allowance for root !where nodes, which are assumed to
            # refer to the script file if the file is omitted.
            file = where.file or default_file
            assert file, "Cannot set line breakpoints without a valid file!"
            # If this Where covers a range of lines, we breakpoint each of them to ensure that we don't miss any lines.
            for line in where.get_lines(self.script.get_labels(file)):
                added_ids.append(self.debugger.add_breakpoint(file, line))
        self._where_bps[where] = added_ids
        for id in added_ids:
            self.context.logger.note(f"Added Entry BP {id} for {where}")

    def _init_bps(self):
        for where in self.script.root_wheres:
            self.add_where_entry_bp(where, self.script.root_scope.file)

    def _run_debugger_custom(self, cmdline):
        if not isinstance(self.debugger, DAP):
            raise NotImplementedError(
                "Only DAP-based debuggers currently supported in structured scripts."
            )

        self.step_collection.clear_steps()

        script: DexterScript = self.script
        state_match_context = StateMatchContext()
        self._init_bps()

        self.debugger.launch(cmdline)
        time.sleep(self._pause_between_steps)

        timed_out = False
        total_timeout = Timeout(self.context.options.timeout_total)

        while not self.debugger.is_finished:
            ## Check for timeouts.
            breakpoint_timeout = Timeout(self.context.options.timeout_breakpoint)
            while self.debugger.is_running and not timed_out:
                # Check to see whether we've timed out while we're waiting.
                if total_timeout.timed_out():
                    self.context.logger.error(
                        "Debugger session has been "
                        f"running for {total_timeout.elapsed}s, timeout reached!"
                    )
                    timed_out = True
                if breakpoint_timeout.timed_out():
                    self.context.logger.error(
                        f"Debugger session has not "
                        f"hit a breakpoint for {breakpoint_timeout.elapsed}s, timeout "
                        "reached!"
                    )
                    timed_out = True

            if timed_out or self.debugger.is_finished:
                break

            ## Fetch frame information and breakpoint information from the debugger.
            step_info: StepIR = self.debugger.get_stack_frames(self._step_index)
            # For some niche cases around function breakpoints, we may fail to
            # correctly notice a !where being hit; therefore we explicitly track
            # !where breakpoints that get hit in the StepIR.
            hit_bps = self.debugger.get_triggered_breakpoint_ids()
            step_info.hit_where_bps = sorted(
                (
                    where
                    for where, bp_ids in self._where_bps.items()
                    if any(bp_id in hit_bps for bp_id in bp_ids)
                ),
                key=lambda where: str(where),
            )

            active_where_matches = get_active_where_matches(
                script, step_info, state_match_context
            )

            watches = [
                watch
                for where_match in active_where_matches.values()
                for expect in where_match.active_expects
                if (watch := expect.get_watched_expr())
            ]
            scope_watches = [
                scope_watch
                for where_match in active_where_matches.values()
                for expect in where_match.active_expects
                if (scope_watch := expect.get_watched_scope())
            ]
            self.debugger.collect_watches(step_info, watches, scope_watches)

            active_thens = [
                then
                for where_match in active_where_matches.values()
                for then in where_match.active_thens
            ]
            should_step_out = any(then.command == "step_out" for then in active_thens)
            should_finish = any(then.command == "finish" for then in active_thens)

            # Our stepping behaviour is as follows:
            # - If any !then is active, we follow its command.
            # - Otherwise, if any !where matches the current stack frame, we step.
            # - Otherwise, if any !where matches any non-current stack frame, we step out.
            # - Otherwise, we continue.
            if should_finish:
                next_action = DebuggerAction.EXIT
            elif should_step_out:
                next_action = DebuggerAction.STEP_OUT
            elif any(
                where_match.frame_idx == 0
                for where_match in active_where_matches.values()
            ):
                next_action = DebuggerAction.STEP_OVER
            elif active_where_matches:
                next_action = DebuggerAction.STEP_OUT
            elif all(
                where in state_match_context.expired_wheres
                for where in script.root_wheres
            ):
                next_action = DebuggerAction.EXIT
            else:
                next_action = DebuggerAction.CONTINUE

            # Update breakpoints: first remove unneeded breakpoints, then set newly desired breakpoints.
            bp_to_delete = []
            pending_wheres = set(
                where
                for where_match in active_where_matches.values()
                for where in where_match.pending_wheres
            )

            def where_should_have_breakpoint(where: Where):
                if where in state_match_context.expired_wheres:
                    return False
                if where in script.root_wheres:
                    return True
                if should_step_out:
                    return False
                return where in pending_wheres
            for where, bp_ids in self._where_bps.items():
                if bp_ids and not where_should_have_breakpoint(where):
                    bp_to_delete.extend(bp_ids)
                    bp_ids.clear()
            self.debugger.delete_breakpoints(bp_to_delete)
            for where in pending_wheres:
                if not self._where_bps[where] and where_should_have_breakpoint(where):
                    self.add_where_entry_bp(where)

            if step_info.current_frame:
                self._step_index += 1
                # Record the step in step_collection.
                self.step_collection.new_step(self.context, step_info)
                if self._step_index > self._max_steps:
                    next_action = DebuggerAction.EXIT

            # If we have --trace enabled, report a short overview of this step.
            self.context.logger.note(
                f"Stopped at {step_info.current_function} {step_info.current_location.short_str()}, {len(active_where_matches)} !wheres on the stack, next_action={next_action}"
            )

            if next_action == DebuggerAction.EXIT:
                break
            elif next_action == DebuggerAction.STEP_OVER:
                self.debugger.step_next()
            elif next_action == DebuggerAction.STEP_OUT:
                self.debugger.step_out()
            else:
                assert (
                    next_action == DebuggerAction.CONTINUE
                ), f"next_action has invalid value {next_action}"
                self.debugger.go()
            time.sleep(self._pause_between_steps)
