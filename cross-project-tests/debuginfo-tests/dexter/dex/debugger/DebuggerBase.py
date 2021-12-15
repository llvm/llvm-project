# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base class for all debugger interface implementations."""

import abc
import os
import sys
import traceback
import unittest

from types import SimpleNamespace
from dex.command.CommandBase import StepExpectInfo
from dex.dextIR import DebuggerIR, FrameIR, LocIR, StepIR, ValueIR
from dex.utils.Exceptions import DebuggerException
from dex.utils.ReturnCode import ReturnCode

def watch_is_active(watch_info: StepExpectInfo, path, frame_idx, line_no):
    _, watch_path, watch_frame_idx, watch_line_range = watch_info
    # If this watch should only be active for a specific file...
    if watch_path and os.path.isfile(watch_path):
        # If the current path does not match the expected file, this watch is
        # not active.
        if not (path and os.path.isfile(path) and
                os.path.samefile(path, watch_path)):
            return False
    if watch_frame_idx != frame_idx:
        return False
    if watch_line_range and line_no not in list(watch_line_range):
        return False
    return True

class DebuggerBase(object, metaclass=abc.ABCMeta):
    def __init__(self, context):
        self.context = context
        # Note: We can't already read values from options
        # as DebuggerBase is created before we initialize options
        # to read potential_debuggers.
        self.options = self.context.options

        self._interface = None
        self.has_loaded = False
        self._loading_error = None
        try:
            self._interface = self._load_interface()
            self.has_loaded = True
        except DebuggerException:
            self._loading_error = sys.exc_info()

    def __enter__(self):
        try:
            self._custom_init()
            self.clear_breakpoints()
        except DebuggerException:
            self._loading_error = sys.exc_info()
        return self

    def __exit__(self, *args):
        self._custom_exit()

    def _custom_init(self):
        pass

    def _custom_exit(self):
        pass

    @property
    def debugger_info(self):
        return DebuggerIR(name=self.name, version=self.version)

    @property
    def is_available(self):
        return self.has_loaded and self.loading_error is None

    @property
    def loading_error(self):
        return (str(self._loading_error[1])
                if self._loading_error is not None else None)

    @property
    def loading_error_trace(self):
        if not self._loading_error:
            return None

        tb = traceback.format_exception(*self._loading_error)

        if self._loading_error[1].orig_exception is not None:
            orig_exception = traceback.format_exception(
                *self._loading_error[1].orig_exception)

            if ''.join(orig_exception) not in ''.join(tb):
                tb.extend(['\n'])
                tb.extend(orig_exception)

        tb = ''.join(tb).splitlines(True)
        return tb

    def _sanitize_function_name(self, name):  # pylint: disable=no-self-use
        """If the function name returned by the debugger needs any post-
        processing to make it fit (for example, if it includes a byte offset),
        do that here.
        """
        return name

    @abc.abstractmethod
    def _load_interface(self):
        pass

    @classmethod
    def get_option_name(cls):
        """Short name that will be used on the command line to specify this
        debugger.
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls):
        """Full name of this debugger."""
        raise NotImplementedError()

    @property
    def name(self):
        return self.__class__.get_name()

    @property
    def option_name(self):
        return self.__class__.get_option_name()

    @abc.abstractproperty
    def version(self):
        pass

    @abc.abstractmethod
    def clear_breakpoints(self):
        pass

    def add_breakpoint(self, file_, line):
        """Returns a unique opaque breakpoint id.

        The ID type depends on the debugger being used, but will probably be
        an int.
        """
        return self._add_breakpoint(self._external_to_debug_path(file_), line)

    @abc.abstractmethod
    def _add_breakpoint(self, file_, line):
        """Returns a unique opaque breakpoint id.
        """
        pass

    def add_conditional_breakpoint(self, file_, line, condition):
        """Returns a unique opaque breakpoint id.

        The ID type depends on the debugger being used, but will probably be
        an int.
        """
        return self._add_conditional_breakpoint(
            self._external_to_debug_path(file_), line, condition)

    @abc.abstractmethod
    def _add_conditional_breakpoint(self, file_, line, condition):
        """Returns a unique opaque breakpoint id.
        """
        pass

    @abc.abstractmethod
    def delete_breakpoint(self, id):
        """Delete a breakpoint by id.

        Raises a KeyError if no breakpoint with this id exists.
        """
        pass

    @abc.abstractmethod
    def get_triggered_breakpoint_ids(self):
        """Returns a set of opaque ids for just-triggered breakpoints.
        """
        pass

    @abc.abstractmethod
    def launch(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def go(self) -> ReturnCode:
        pass

    def get_step_info(self, watches, step_index):
        step_info = self._get_step_info(watches, step_index)
        for frame in step_info.frames:
            frame.loc.path = self._debug_to_external_path(frame.loc.path)
        return step_info

    @abc.abstractmethod
    def _get_step_info(self, watches, step_index):
        pass

    @abc.abstractproperty
    def is_running(self):
        pass

    @abc.abstractproperty
    def is_finished(self):
        pass

    @abc.abstractproperty
    def frames_below_main(self):
        pass

    @abc.abstractmethod
    def evaluate_expression(self, expression, frame_idx=0) -> ValueIR:
        pass

    def _external_to_debug_path(self, path):
        if not self.options.debugger_use_relative_paths:
            return path
        root_dir = self.options.source_root_dir
        if not root_dir or not path:
            return path
        assert path.startswith(root_dir)
        return path[len(root_dir):].lstrip(os.path.sep)

    def _debug_to_external_path(self, path):
        if not self.options.debugger_use_relative_paths:
            return path
        if not path or not self.options.source_root_dir:
            return path
        for file in self.options.source_files:
            if path.endswith(self._external_to_debug_path(file)):
                return file
        return path

class TestDebuggerBase(unittest.TestCase):

    class MockDebugger(DebuggerBase):

        def __init__(self, context, *args):
            super().__init__(context, *args)
            self.step_info = None
            self.breakpoint_file = None

        def _add_breakpoint(self, file, line):
            self.breakpoint_file = file

        def _get_step_info(self, watches, step_index):
            return self.step_info

    def __init__(self, *args):
        super().__init__(*args)
        TestDebuggerBase.MockDebugger.__abstractmethods__ = set()
        self.options = SimpleNamespace(source_root_dir = '', source_files = [])
        context = SimpleNamespace(options = self.options)
        self.dbg = TestDebuggerBase.MockDebugger(context)

    def _new_step(self, paths):
        frames = [
            FrameIR(
                function=None,
                is_inlined=False,
                loc=LocIR(path=path, lineno=0, column=0)) for path in paths
        ]
        return StepIR(step_index=0, stop_reason=None, frames=frames)

    def _step_paths(self, step):
        return [frame.loc.path for frame in step.frames]

    def test_add_breakpoint_no_source_root_dir(self):
        self.options.debugger_use_relative_paths = True
        self.options.source_root_dir = ''
        path = os.path.join(os.path.sep + 'root', 'some_file')
        self.dbg.add_breakpoint(path, 12)
        self.assertEqual(path, self.dbg.breakpoint_file)

    def test_add_breakpoint_with_source_root_dir(self):
        self.options.debugger_use_relative_paths = True
        self.options.source_root_dir = os.path.sep + 'my_root'
        path = os.path.join(self.options.source_root_dir, 'some_file')
        self.dbg.add_breakpoint(path, 12)
        self.assertEqual('some_file', self.dbg.breakpoint_file)

    def test_add_breakpoint_with_source_root_dir_slash_suffix(self):
        self.options.debugger_use_relative_paths = True
        self.options.source_root_dir = os.path.sep + 'my_root' + os.path.sep
        path = os.path.join(self.options.source_root_dir, 'some_file')
        self.dbg.add_breakpoint(path, 12)
        self.assertEqual('some_file', self.dbg.breakpoint_file)

    def test_get_step_info_no_source_root_dir(self):
        self.options.debugger_use_relative_paths = True
        path = os.path.join(os.path.sep + 'root', 'some_file')
        self.dbg.step_info = self._new_step([path])
        self.assertEqual([path],
            self._step_paths(self.dbg.get_step_info([], 0)))

    def test_get_step_info_no_frames(self):
        self.options.debugger_use_relative_paths = True
        self.options.source_root_dir = os.path.sep + 'my_root'
        self.dbg.step_info = self._new_step([])
        self.assertEqual([],
            self._step_paths(self.dbg.get_step_info([], 0)))

    def test_get_step_info(self):
        self.options.debugger_use_relative_paths = True
        self.options.source_root_dir = os.path.sep + 'my_root'
        path = os.path.join(self.options.source_root_dir, 'some_file')
        self.options.source_files = [path]
        other_path = os.path.join(os.path.sep + 'other', 'file')
        dbg_path = os.path.join(os.path.sep + 'dbg', 'some_file')
        self.dbg.step_info = self._new_step(
            [None, other_path, dbg_path])
        self.assertEqual([None, other_path, path],
            self._step_paths(self.dbg.get_step_info([], 0)))
