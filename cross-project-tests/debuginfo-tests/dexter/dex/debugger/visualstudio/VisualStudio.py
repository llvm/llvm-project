# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Interface for communicating with the Visual Studio debugger via DTE."""

import abc
import imp
import os
import sys
from pathlib import PurePath, Path
from collections import defaultdict, namedtuple

from dex.command.CommandBase import StepExpectInfo
from dex.debugger.DebuggerBase import DebuggerBase, watch_is_active
from dex.dextIR import FrameIR, LocIR, StepIR, StopReason, ValueIR
from dex.dextIR import StackFrame, SourceLocation, ProgramState
from dex.utils.Exceptions import Error, LoadDebuggerException
from dex.utils.ReturnCode import ReturnCode


def _load_com_module():
    try:
        module_info = imp.find_module(
            "ComInterface", [os.path.join(os.path.dirname(__file__), "windows")]
        )
        return imp.load_module("ComInterface", *module_info)
    except ImportError as e:
        raise LoadDebuggerException(e, sys.exc_info())


# VSBreakpoint(path: PurePath, line: int, col: int, cond: str).  This is enough
# info to identify breakpoint equivalence in visual studio based on the
# properties we set through dexter currently.
VSBreakpoint = namedtuple("VSBreakpoint", "path, line, col, cond")


class VisualStudio(
    DebuggerBase, metaclass=abc.ABCMeta
):  # pylint: disable=abstract-method

    # Constants for results of Debugger.CurrentMode
    # (https://msdn.microsoft.com/en-us/library/envdte.debugger.currentmode.aspx)
    dbgDesignMode = 1
    dbgBreakMode = 2
    dbgRunMode = 3

    def __init__(self, *args):
        self.com_module = None
        self._debugger = None
        self._solution = None
        self._fn_step = None
        self._fn_go = None
        # The next available unique breakpoint id. Use self._get_next_id().
        self._next_bp_id = 0
        # VisualStudio appears to common identical breakpoints. That is, if you
        # ask for a breakpoint that already exists the Breakpoints list will
        # not grow. DebuggerBase requires all breakpoints have a unique id,
        # even for duplicates, so we'll need to do some bookkeeping.  Map
        # {VSBreakpoint: list(id)} where id is the unique dexter-side id for
        # the requested breakpoint.
        self._vs_to_dex_ids = defaultdict(list)
        # Map {id: VSBreakpoint} where id is unique and VSBreakpoint identifies
        # a breakpoint in Visual Studio. There may be many ids mapped to a
        # single VSBreakpoint. Use self._vs_to_dex_ids to find (dexter)
        # breakpoints mapped to the same visual studio breakpoint.
        self._dex_id_to_vs = {}

        super(VisualStudio, self).__init__(*args)

    def _create_solution(self):
        self._solution.Create(self.context.working_directory.path, "DexterSolution")
        try:
            self._solution.AddFromFile(self._project_file)
        except OSError:
            raise LoadDebuggerException(
                "could not debug the specified executable", sys.exc_info()
            )

    def _load_solution(self):
        try:
            self._solution.Open(self.context.options.vs_solution)
        except:
            raise LoadDebuggerException(
                "could not load specified vs solution at {}".format(
                    self.context.options.vs_solution
                ),
                sys.exc_info(),
            )

    def _custom_init(self):
        try:
            self._debugger = self._interface.Debugger
            self._debugger.HexDisplayMode = False

            self._interface.MainWindow.Visible = self.context.options.show_debugger

            self._solution = self._interface.Solution
            if self.context.options.vs_solution is None:
                self._create_solution()
            else:
                self._load_solution()

            self._fn_step = self._debugger.StepInto
            self._fn_go = self._debugger.Go

        except AttributeError as e:
            raise LoadDebuggerException(str(e), sys.exc_info())

    def _custom_exit(self):
        if self._interface:
            self._interface.Quit()

    @property
    def _project_file(self):
        return self.context.options.executable

    @abc.abstractproperty
    def _dte_version(self):
        pass

    @property
    def _location(self):
        # TODO: Find a better way of determining path, line and column info
        # that doesn't require reading break points. This method requires
        # all lines to have a break point on them.
        bp = self._debugger.BreakpointLastHit
        return {
            "path": getattr(bp, "File", None),
            "lineno": getattr(bp, "FileLine", None),
            "column": getattr(bp, "FileColumn", None),
        }

    @property
    def _mode(self):
        return self._debugger.CurrentMode

    def _load_interface(self):
        self.com_module = _load_com_module()
        return self.com_module.DTE(self._dte_version)

    @property
    def version(self):
        try:
            return self._interface.Version
        except AttributeError:
            return None

    def clear_breakpoints(self):
        for bp in self._debugger.Breakpoints:
            bp.Delete()
        self._vs_to_dex_ids.clear()
        self._dex_id_to_vs.clear()

    def _add_breakpoint(self, file_, line):
        return self._add_conditional_breakpoint(file_, line, "")

    def _get_next_id(self):
        # "Generate" a new unique id for the breakpoint.
        id = self._next_bp_id
        self._next_bp_id += 1
        return id

    def _add_conditional_breakpoint(self, file_, line, condition):
        col = 1
        vsbp = VSBreakpoint(PurePath(file_), line, col, condition)
        new_id = self._get_next_id()

        # Do we have an exact matching breakpoint already?
        if vsbp in self._vs_to_dex_ids:
            self._vs_to_dex_ids[vsbp].append(new_id)
            self._dex_id_to_vs[new_id] = vsbp
            return new_id

        # Breakpoint doesn't exist already. Add it now.
        count_before = self._debugger.Breakpoints.Count
        self._debugger.Breakpoints.Add("", file_, line, col, condition)
        # Our internal representation of VS says that the breakpoint doesn't
        # already exist so we do not expect this operation to fail here.
        assert count_before < self._debugger.Breakpoints.Count
        # We've added a new breakpoint, record its id.
        self._vs_to_dex_ids[vsbp].append(new_id)
        self._dex_id_to_vs[new_id] = vsbp
        return new_id

    def get_triggered_breakpoint_ids(self):
        """Returns a set of opaque ids for just-triggered breakpoints."""
        bps_hit = self._debugger.AllBreakpointsLastHit
        bp_id_list = []
        # Intuitively, AllBreakpointsLastHit breakpoints are the last hit
        # _bound_ breakpoints. A bound breakpoint's parent holds the info of
        # the breakpoint the user requested. Our internal state tracks the user
        # requested breakpoints so we look at the Parent of these triggered
        # breakpoints to determine which have been hit.
        for bp in bps_hit:
            # All bound breakpoints should have the user-defined breakpoint as
            # a parent.
            assert bp.Parent
            vsbp = VSBreakpoint(
                PurePath(bp.Parent.File),
                bp.Parent.FileLine,
                bp.Parent.FileColumn,
                bp.Parent.Condition,
            )
            try:
                ids = self._vs_to_dex_ids[vsbp]
            except KeyError:
                pass
            else:
                bp_id_list += ids
        return set(bp_id_list)

    def delete_breakpoints(self, ids):
        """Delete breakpoints by their ids.

        Raises a KeyError if no breakpoint with this id exists.
        """
        vsbp_set = set()
        for id in ids:
            vsbp = self._dex_id_to_vs[id]

            # Remove our id from the associated list of dex ids.
            self._vs_to_dex_ids[vsbp].remove(id)
            del self._dex_id_to_vs[id]

            # Bail if there are other uses of this vsbp.
            if len(self._vs_to_dex_ids[vsbp]) > 0:
                continue
            # Otherwise find and delete it.
            vsbp_set.add(vsbp)

        vsbp_to_del_count = len(vsbp_set)

        for bp in self._debugger.Breakpoints:
            # We're looking at the user-set breakpoints so there should be no
            # Parent.
            assert bp.Parent == None
            this_vsbp = VSBreakpoint(
                PurePath(bp.File), bp.FileLine, bp.FileColumn, bp.Condition
            )
            if this_vsbp in vsbp_set:
                bp.Delete()
                vsbp_to_del_count -= 1
                if vsbp_to_del_count == 0:
                    break
        if vsbp_to_del_count:
            raise KeyError("did not find breakpoint to be deleted")

    def _fetch_property(self, props, name):
        num_props = props.Count
        result = None
        for x in range(1, num_props + 1):
            item = props.Item(x)
            if item.Name == name:
                return item
        assert False, "Couldn't find property {}".format(name)

    def launch(self, cmdline):
        exe_path = Path(self.context.options.executable)
        self.context.logger.note(f"VS: Using executable: '{exe_path}'")
        cmdline_str = " ".join(cmdline)
        if self.context.options.target_run_args:
            cmdline_str += f" {self.context.options.target_run_args}"
        if cmdline_str:
            self.context.logger.note(f"VS: Using executable args: '{cmdline_str}'")

        # In a slightly baroque manner, lookup the VS project that runs when
        # you click "run", and set its command line options to the desired
        # command line options.
        startup_proj_name = str(
            self._fetch_property(self._interface.Solution.Properties, "StartupProject")
        )
        project = self._fetch_property(self._interface.Solution, startup_proj_name)
        ActiveConfiguration = self._fetch_property(
            project.Properties, "ActiveConfiguration"
        ).Object
        ActiveConfiguration.DebugSettings.CommandArguments = cmdline_str

        self.context.logger.note("Launching VS debugger...")
        self._fn_go(False)

    def step(self):
        self._fn_step(False)

    def go(self) -> ReturnCode:
        self._fn_go(False)
        return ReturnCode.OK

    def set_current_stack_frame(self, idx: int = 0):
        thread = self._debugger.CurrentThread
        stack_frames = thread.StackFrames
        try:
            stack_frame = stack_frames[idx]
            self._debugger.CurrentStackFrame = stack_frame.raw
        except IndexError:
            raise Error(
                "attempted to access stack frame {} out of {}".format(
                    idx, len(stack_frames)
                )
            )

    def _get_step_info(self, watches, step_index):
        thread = self._debugger.CurrentThread
        stackframes = thread.StackFrames

        frames = []
        state_frames = []

        loc = LocIR(**self._location)
        valid_loc_for_watch = loc.path and os.path.exists(loc.path)

        for idx, sf in enumerate(stackframes):
            frame = FrameIR(
                function=self._sanitize_function_name(sf.FunctionName),
                is_inlined=sf.FunctionName.startswith("[Inline Frame]"),
                loc=LocIR(path=None, lineno=None, column=None),
            )

            fname = frame.function or ""  # pylint: disable=no-member
            if any(name in fname for name in self.frames_below_main):
                break

            state_frame = StackFrame(
                function=frame.function, is_inlined=frame.is_inlined, watches={}
            )

            if valid_loc_for_watch and idx == 0:
                for watch_info in watches:
                    if watch_is_active(watch_info, loc.path, idx, loc.lineno):
                        watch_expr = watch_info.expression
                        state_frame.watches[watch_expr] = self.evaluate_expression(
                            watch_expr, idx
                        )

            state_frames.append(state_frame)
            frames.append(frame)

        if frames:
            frames[0].loc = loc
            state_frames[0].location = SourceLocation(**self._location)

        reason = StopReason.BREAKPOINT
        if loc.path is None:  # pylint: disable=no-member
            reason = StopReason.STEP

        program_state = ProgramState(frames=state_frames)

        return StepIR(
            step_index=step_index,
            frames=frames,
            stop_reason=reason,
            program_state=program_state,
        )

    @property
    def is_running(self):
        return self._mode == VisualStudio.dbgRunMode

    @property
    def is_finished(self):
        return self._mode == VisualStudio.dbgDesignMode

    @property
    def frames_below_main(self):
        return [
            "[Inline Frame] invoke_main",
            "__scrt_common_main_seh",
            "__tmainCRTStartup",
            "mainCRTStartup",
        ]

    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        if frame_idx != 0:
            self.set_current_stack_frame(frame_idx)
        result = self._debugger.GetExpression(expression)
        if frame_idx != 0:
            self.set_current_stack_frame(0)
        value = result.Value

        is_optimized_away = any(
            s in value
            for s in [
                "Variable is optimized away and not available",
                "Value is not available, possibly due to optimization",
            ]
        )

        is_irretrievable = any(
            s in value
            for s in [
                "???",
                "<Unable to read memory>",
            ]
        )

        # an optimized away value is still counted as being able to be
        # evaluated.
        could_evaluate = result.IsValidValue or is_optimized_away or is_irretrievable

        return ValueIR(
            expression=expression,
            value=value,
            type_name=result.Type,
            error_string=None,
            is_optimized_away=is_optimized_away,
            could_evaluate=could_evaluate,
            is_irretrievable=is_irretrievable,
        )
