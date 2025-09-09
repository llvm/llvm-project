# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interface for communicating with the LLDB debugger via its python interface.
"""

import os
import shlex
from subprocess import CalledProcessError, check_output, STDOUT
import sys

from dex.debugger.DebuggerBase import DebuggerBase, watch_is_active
from dex.debugger.DAP import DAP
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import StackFrame, SourceLocation, ProgramState
from dex.utils.Exceptions import DebuggerException, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode
from dex.utils.Imports import load_module


class LLDB(DebuggerBase):
    def __init__(self, context, *args):
        self.lldb_executable = context.options.lldb_executable
        self._debugger = None
        self._target = None
        self._process = None
        self._thread = None
        # Map {id (int): condition (str)} for breakpoints which have a
        # condition. See get_triggered_breakpoint_ids usage for more info.
        self._breakpoint_conditions = {}
        super(LLDB, self).__init__(context, *args)

    def _custom_init(self):
        self._debugger = self._interface.SBDebugger.Create()
        self._debugger.SetAsync(False)
        self._target = self._debugger.CreateTargetWithFileAndArch(
            self.context.options.executable, self.context.options.arch
        )
        if not self._target:
            raise LoadDebuggerException(
                'could not create target for executable "{}" with arch:{}'.format(
                    self.context.options.executable, self.context.options.arch
                )
            )

    def _custom_exit(self):
        if getattr(self, "_process", None):
            self._process.Kill()
        if getattr(self, "_debugger", None) and getattr(self, "_target", None):
            self._debugger.DeleteTarget(self._target)

    def _translate_stop_reason(self, reason):
        if reason == self._interface.eStopReasonNone:
            return None
        if reason == self._interface.eStopReasonBreakpoint:
            return StopReason.BREAKPOINT
        if reason == self._interface.eStopReasonPlanComplete:
            return StopReason.STEP
        if reason == self._interface.eStopReasonThreadExiting:
            return StopReason.PROGRAM_EXIT
        if reason == self._interface.eStopReasonException:
            return StopReason.ERROR
        return StopReason.OTHER

    def _load_interface(self):
        try:
            args = [self.lldb_executable, "-P"]
            pythonpath = check_output(args, stderr=STDOUT).rstrip().decode("utf-8")
        except CalledProcessError as e:
            raise LoadDebuggerException(str(e), sys.exc_info())
        except OSError as e:
            raise LoadDebuggerException(
                '{} ["{}"]'.format(e.strerror, self.lldb_executable), sys.exc_info()
            )

        if not os.path.isdir(pythonpath):
            raise LoadDebuggerException(
                'path "{}" does not exist [result of {}]'.format(pythonpath, args),
                sys.exc_info(),
            )

        try:
            return load_module("lldb", pythonpath)
        except ImportError as e:
            msg = str(e)
            if msg.endswith("not a valid Win32 application."):
                msg = "{} [Are you mixing 32-bit and 64-bit binaries?]".format(msg)
            raise LoadDebuggerException(msg, sys.exc_info())

    @classmethod
    def get_name(cls):
        return "lldb"

    @classmethod
    def get_option_name(cls):
        return "lldb"

    @property
    def version(self):
        try:
            return self._interface.SBDebugger_GetVersionString()
        except AttributeError:
            return None

    def clear_breakpoints(self):
        self._target.DeleteAllBreakpoints()

    def _add_breakpoint(self, file_, line):
        return self._add_conditional_breakpoint(file_, line, None)

    def _add_conditional_breakpoint(self, file_, line, condition):
        bp = self._target.BreakpointCreateByLocation(file_, line)
        if not bp:
            raise DebuggerException(
                "could not add breakpoint [{}:{}]".format(file_, line)
            )
        id = bp.GetID()
        if condition:
            bp.SetCondition(condition)
            assert id not in self._breakpoint_conditions
            self._breakpoint_conditions[id] = condition
        return id

    def _evaulate_breakpoint_condition(self, id):
        """Evaluate the breakpoint condition and return the result.

        Returns True if a conditional breakpoint with the specified id cannot
        be found (i.e. assume it is an unconditional breakpoint).
        """
        try:
            condition = self._breakpoint_conditions[id]
        except KeyError:
            # This must be an unconditional breakpoint.
            return True
        valueIR = self.evaluate_expression(condition)
        return valueIR.type_name == "bool" and valueIR.value == "true"

    def get_triggered_breakpoint_ids(self):
        # Breakpoints can only have been triggered if we've hit one.
        stop_reason = self._translate_stop_reason(self._thread.GetStopReason())
        if stop_reason != StopReason.BREAKPOINT:
            return []
        breakpoint_ids = set()
        # When the stop reason is eStopReasonBreakpoint, GetStopReasonDataCount
        # counts all breakpoints associated with the location that lldb has
        # stopped at, regardless of their condition. I.e. Even if we have two
        # breakpoints at the same source location that have mutually exclusive
        # conditions, both will be counted by GetStopReasonDataCount when
        # either condition is true. Check each breakpoint condition manually to
        # filter the list down to breakpoints that have caused this stop.
        #
        # Breakpoints have two data parts: Breakpoint ID, Location ID. We're
        # only interested in the Breakpoint ID so we skip every other item.
        for i in range(0, self._thread.GetStopReasonDataCount(), 2):
            id = self._thread.GetStopReasonDataAtIndex(i)
            if self._evaulate_breakpoint_condition(id):
                breakpoint_ids.add(id)
        return breakpoint_ids

    def delete_breakpoints(self, ids):
        for id in ids:
            bp = self._target.FindBreakpointByID(id)
            if not bp:
                # The ID is not valid.
                raise KeyError
            try:
                del self._breakpoint_conditions[id]
            except KeyError:
                # This must be an unconditional breakpoint.
                pass
            self._target.BreakpointDelete(id)

    def launch(self, cmdline):
        num_resolved_breakpoints = 0
        for b in self._target.breakpoint_iter():
            num_resolved_breakpoints += b.GetNumLocations() > 0
        assert num_resolved_breakpoints > 0

        if self.context.options.target_run_args:
            cmdline += shlex.split(self.context.options.target_run_args)
        launch_info = self._target.GetLaunchInfo()
        launch_info.SetWorkingDirectory(os.getcwd())
        launch_info.SetArguments(cmdline, True)
        error = self._interface.SBError()
        self._process = self._target.Launch(launch_info, error)
        
        if error.Fail():
            raise DebuggerException(error.GetCString())
        if not os.path.exists(self._target.executable.fullpath):
            raise DebuggerException("exe does not exist")
        if not self._process or self._process.GetNumThreads() == 0:
            raise DebuggerException("could not launch process")
        if self._process.GetNumThreads() != 1:
            raise DebuggerException("multiple threads not supported")
        self._thread = self._process.GetThreadAtIndex(0)
        
        num_stopped_threads = 0
        for thread in self._process:
            if thread.GetStopReason() == self._interface.eStopReasonBreakpoint:
                num_stopped_threads += 1
        assert num_stopped_threads > 0
        assert self._thread, (self._process, self._thread)

    def step_in(self):
        self._thread.StepInto()
        stop_reason = self._thread.GetStopReason()
        # If we (1) completed a step and (2) are sitting at a breakpoint,
        # but (3) the breakpoint is not reported as the stop reason, then
        # we'll need to step once more to hit the breakpoint.
        #
        # dexter sets breakpoints on every source line, then steps
        # each source line. Older lldb's would overwrite the stop
        # reason with "breakpoint hit" when we stopped at a breakpoint,
        # even if the breakpoint hadn't been exectued yet.  One
        # step per source line, hitting a breakpoint each time.
        #
        # But a more accurate behavior is that the step completes
        # with step-completed stop reason, then when we step again,
        # we execute the breakpoint and stop (with the pc the same) and
        # a breakpoint-hit stop reason.  So we need to step twice per line.
        if stop_reason == self._interface.eStopReasonPlanComplete:
            stepped_to_breakpoint = False
            pc = self._thread.GetFrameAtIndex(0).GetPC()
            for bp in self._target.breakpoints:
                for bploc in bp.locations:
                    if (
                        bploc.IsEnabled()
                        and bploc.GetAddress().GetLoadAddress(self._target) == pc
                    ):
                        stepped_to_breakpoint = True
            if stepped_to_breakpoint:
                self._process.Continue()

    def go(self) -> ReturnCode:
        self._process.Continue()
        return ReturnCode.OK

    def _get_step_info(self, watches, step_index):
        frames = []
        state_frames = []

        for i in range(0, self._thread.GetNumFrames()):
            sb_frame = self._thread.GetFrameAtIndex(i)
            sb_line = sb_frame.GetLineEntry()
            sb_filespec = sb_line.GetFileSpec()

            try:
                path = os.path.join(
                    sb_filespec.GetDirectory(), sb_filespec.GetFilename()
                )
            except (AttributeError, TypeError):
                path = None

            function = self._sanitize_function_name(sb_frame.GetFunctionName())

            loc_dict = {
                "path": path,
                "lineno": sb_line.GetLine(),
                "column": sb_line.GetColumn(),
            }
            loc = LocIR(**loc_dict)
            valid_loc_for_watch = loc.path and os.path.exists(loc.path)

            frame = FrameIR(function=function, is_inlined=sb_frame.IsInlined(), loc=loc)

            if any(
                name in (frame.function or "")  # pylint: disable=no-member
                for name in self.frames_below_main
            ):
                break

            frames.append(frame)

            state_frame = StackFrame(
                function=frame.function,
                is_inlined=frame.is_inlined,
                location=SourceLocation(**loc_dict),
                watches={},
            )
            if valid_loc_for_watch:
                for expr in map(
                    # Filter out watches that are not active in the current frame,
                    # and then evaluate all the active watches.
                    lambda watch_info, idx=i: self.evaluate_expression(
                        watch_info.expression, idx
                    ),
                    filter(
                        lambda watch_info, idx=i, line_no=loc.lineno, loc_path=loc.path: watch_is_active(
                            watch_info, loc_path, idx, line_no
                        ),
                        watches,
                    ),
                ):
                    state_frame.watches[expr.expression] = expr
            state_frames.append(state_frame)

        if len(frames) == 1 and frames[0].function is None:
            frames = []
            state_frames = []

        reason = self._translate_stop_reason(self._thread.GetStopReason())

        return StepIR(
            step_index=step_index,
            frames=frames,
            stop_reason=reason,
            program_state=ProgramState(state_frames),
        )

    @property
    def is_running(self):
        # We're not running in async mode so this is always False.
        return False

    @property
    def is_finished(self):
        return not self._thread.GetFrameAtIndex(0)

    @property
    def frames_below_main(self):
        return ["__scrt_common_main_seh", "__libc_start_main", "__libc_start_call_main"]

    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        result = self._thread.GetFrameAtIndex(frame_idx).EvaluateExpression(expression)
        error_string = str(result.error)

        value = result.value
        could_evaluate = not any(
            s in error_string
            for s in [
                "Can't run the expression locally",
                "use of undeclared identifier",
                "no member named",
                "Couldn't lookup symbols",
                "Couldn't look up symbols",
                "reference to local variable",
                "invalid use of 'this' outside of a non-static member function",
            ]
        )

        is_optimized_away = any(
            s in error_string
            for s in [
                "value may have been optimized out",
            ]
        )

        is_irretrievable = any(
            s in error_string
            for s in [
                "couldn't get the value of variable",
                "couldn't read its memory",
                "couldn't read from memory",
                "Cannot access memory at address",
                "invalid address (fault address:",
            ]
        )

        if could_evaluate and not is_irretrievable and not is_optimized_away:
            assert error_string == "success", (error_string, expression, value)
            # assert result.value is not None, (result.value, expression)

        if error_string == "success":
            error_string = None

        # attempt to find expression as a variable, if found, take the variable
        # obj's type information as it's 'usually' more accurate.
        var_result = self._thread.GetFrameAtIndex(frame_idx).FindVariable(expression)
        if str(var_result.error) == "success":
            type_name = var_result.type.GetDisplayTypeName()
        else:
            type_name = result.type.GetDisplayTypeName()

        return ValueIR(
            expression=expression,
            value=value,
            type_name=type_name,
            error_string=error_string,
            could_evaluate=could_evaluate,
            is_optimized_away=is_optimized_away,
            is_irretrievable=is_irretrievable,
        )


class LLDBDAP(DAP):
    def __init__(self, context, *args):
        self.lldb_dap_executable = context.options.lldb_executable
        super(LLDBDAP, self).__init__(context, *args)

    @classmethod
    def get_name(cls):
        return "lldb-dap"

    @classmethod
    def get_option_name(cls):
        return "lldb-dap"

    @property
    def version(self):
        return 1

    @property
    def _debug_adapter_name(self) -> str:
        return "lldb-dap"

    @property
    def _debug_adapter_executable(self) -> str:
        return self.lldb_dap_executable

    @property
    def frames_below_main(self):
        return [
            "__scrt_common_main_seh",
            "__libc_start_main",
            "__libc_start_call_main",
            "_start",
        ]

    def _post_step_hook(self):
        """Hook to be executed after completing a step request."""
        if self._debugger_state.stopped_reason == "step":
            trace_req_id = self.send_message(
                self.make_request(
                    "stackTrace", {"threadId": self._debugger_state.thread, "levels": 1}
                )
            )
            trace_response = self._await_response(trace_req_id)
            if not trace_response["success"]:
                raise DebuggerException("failed to get stack frames")
            try:
                stackframes = trace_response["body"]["stackFrames"]
                path = stackframes[0]["source"]["path"]
                addr = stackframes[0]["instructionPointerReference"]
            except KeyError as e:
                # Temporarily print the DAP log if this fails to aid debugging
                # a buildbot failure that doesn't reproduce easily.
                print(self.message_logger.text.getvalue(), file=sys.stderr)
                raise e

            if any(
                self._debugger_state.bp_addr_map.get(self.dex_id_to_dap_id[dex_bp_id])
                == addr
                for dex_bp_id in self.file_to_bp.get(path, [])
            ):
                # Step again now to get to the breakpoint.
                step_req_id = self.send_message(
                    self.make_request(
                        "continue", {"threadId": self._debugger_state.thread}
                    )
                )
                response = self._await_response(step_req_id)
                if not response["success"]:
                    raise DebuggerException("failed to step")

    def _get_launch_params(self, cmdline):
        cwd = os.getcwd()
        return {
            "cwd": cwd,
            "args": cmdline,
            "program": self.context.options.executable,
        }

    @staticmethod
    def _evaluate_result_value(
        expression: str, result_string: str, type_string
    ) -> ValueIR:
        could_evaluate = not any(
            s in result_string
            for s in [
                "Can't run the expression locally",
                "use of undeclared identifier",
                "no member named",
                "Couldn't lookup symbols",
                "Couldn't look up symbols",
                "reference to local variable",
                "invalid use of 'this' outside of a non-static member function",
            ]
        )

        is_optimized_away = any(
            s in result_string
            for s in [
                "value may have been optimized out",
            ]
        )

        is_irretrievable = any(
            s in result_string
            for s in [
                "couldn't get the value of variable",
                "couldn't read its memory",
                "couldn't read from memory",
                "Cannot access memory at address",
                "invalid address (fault address:",
            ]
        )

        if could_evaluate and not is_irretrievable and not is_optimized_away:
            error_string = None
        else:
            error_string = result_string

        return ValueIR(
            expression=expression,
            value=result_string,
            type_name=type_string,
            error_string=error_string,
            could_evaluate=could_evaluate,
            is_optimized_away=is_optimized_away,
            is_irretrievable=is_irretrievable,
        )

    def _update_requested_bp_list(self, bp_list):
        """ "As lldb-dap cannot have multiple breakpoints at the same location with different conditions, we must
        manually merge conditions here."""
        line_to_cond = {}
        for bp in bp_list:
            if bp.condition is None:
                line_to_cond[bp.line] = None
                continue
            # If we have a condition, we merge it with the existing condition if one exists, unless the known condition
            # is None in which case we preserve the None condition (as the underlying breakpoint should always be hit).
            if bp.line not in line_to_cond:
                line_to_cond[bp.line] = f"({bp.condition})"
            elif line_to_cond[bp.line] is not None:
                line_to_cond[bp.line] = f"{line_to_cond[bp.line]} || ({bp.condition})"
            bp.condition = line_to_cond[bp.line]
        return bp_list

    def _confirm_triggered_breakpoint_ids(self, dex_bp_ids):
        """ "As lldb returns every breakpoint at the current PC regardless of whether their condition was met, we must
        manually check conditions here."""
        confirmed_breakpoint_ids = set()
        for dex_bp_id in dex_bp_ids:
            # Function and instruction breakpoints don't use conditions.
            # FIXME: That's not a DAP restriction, so they could in future.
            if dex_bp_id not in self.bp_info:
                assert (
                    dex_bp_id in self.function_bp_info
                    or dex_bp_id in self.instruction_bp_info
                )
                confirmed_breakpoint_ids.add(dex_bp_id)
                continue

            _, _, cond = self.bp_info[dex_bp_id]
            if cond is None:
                confirmed_breakpoint_ids.add(dex_bp_id)
                continue
            valueIR = self.evaluate_expression(cond)
            if valueIR.type_name == "bool" and valueIR.value == "true":
                confirmed_breakpoint_ids.add(dex_bp_id)
        return confirmed_breakpoint_ids
