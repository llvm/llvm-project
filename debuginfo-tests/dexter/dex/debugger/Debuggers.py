# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Discover potential/available debugger interfaces."""

from collections import OrderedDict
import os
import pickle
import subprocess
import sys
from tempfile import NamedTemporaryFile

from dex.command import find_all_commands
from dex.dextIR import DextIR
from dex.utils import get_root_directory, Timer
from dex.utils.Environment import is_native_windows
from dex.utils.Exceptions import CommandParseError, DebuggerException
from dex.utils.Exceptions import ToolArgumentError
from dex.utils.Warning import warn

from dex.debugger.dbgeng.dbgeng import DbgEng
from dex.debugger.lldb.LLDB import LLDB
from dex.debugger.visualstudio.VisualStudio2015 import VisualStudio2015
from dex.debugger.visualstudio.VisualStudio2017 import VisualStudio2017


def _get_potential_debuggers():  # noqa
    """Return a dict of the supported debuggers.
    Returns:
        { name (str): debugger (class) }
    """
    return {
        DbgEng.get_option_name(): DbgEng,
        LLDB.get_option_name(): LLDB,
        VisualStudio2015.get_option_name(): VisualStudio2015,
        VisualStudio2017.get_option_name(): VisualStudio2017
    }


def _warn_meaningless_option(context, option):
    if context.options.list_debuggers:
        return

    warn(context,
         'option <y>"{}"</> is meaningless with this debugger'.format(option),
         '--debugger={}'.format(context.options.debugger))


def add_debugger_tool_base_arguments(parser, defaults):
    defaults.lldb_executable = 'lldb.exe' if is_native_windows() else 'lldb'
    parser.add_argument(
        '--lldb-executable',
        type=str,
        metavar='<file>',
        default=None,
        display_default=defaults.lldb_executable,
        help='location of LLDB executable')


def add_debugger_tool_arguments(parser, context, defaults):
    debuggers = Debuggers(context)
    potential_debuggers = sorted(debuggers.potential_debuggers().keys())

    add_debugger_tool_base_arguments(parser, defaults)

    parser.add_argument(
        '--debugger',
        type=str,
        choices=potential_debuggers,
        required=True,
        help='debugger to use')
    parser.add_argument(
        '--max-steps',
        metavar='<int>',
        type=int,
        default=1000,
        help='maximum number of program steps allowed')
    parser.add_argument(
        '--pause-between-steps',
        metavar='<seconds>',
        type=float,
        default=0.0,
        help='number of seconds to pause between steps')
    defaults.show_debugger = False
    parser.add_argument(
        '--show-debugger',
        action='store_true',
        default=None,
        help='show the debugger')
    defaults.arch = 'x86_64'
    parser.add_argument(
        '--arch',
        type=str,
        metavar='<architecture>',
        default=None,
        display_default=defaults.arch,
        help='target architecture')


def handle_debugger_tool_base_options(context, defaults):  # noqa
    options = context.options

    if options.lldb_executable is None:
        options.lldb_executable = defaults.lldb_executable
    else:
        if getattr(options, 'debugger', 'lldb') != 'lldb':
            _warn_meaningless_option(context, '--lldb-executable')

        options.lldb_executable = os.path.abspath(options.lldb_executable)
        if not os.path.isfile(options.lldb_executable):
            raise ToolArgumentError('<d>could not find</> <r>"{}"</>'.format(
                options.lldb_executable))


def handle_debugger_tool_options(context, defaults):  # noqa
    options = context.options

    handle_debugger_tool_base_options(context, defaults)

    if options.arch is None:
        options.arch = defaults.arch
    else:
        if options.debugger != 'lldb':
            _warn_meaningless_option(context, '--arch')

    if options.show_debugger is None:
        options.show_debugger = defaults.show_debugger
    else:
        if options.debugger == 'lldb':
            _warn_meaningless_option(context, '--show-debugger')


def _get_command_infos(context):
    commands = find_all_commands(context.options.source_files)
    command_infos = OrderedDict()
    for command_type in commands:
        for command in commands[command_type].values():
            if command_type not in command_infos:
                command_infos[command_type] = []
            command_infos[command_type].append(command)
    return OrderedDict(command_infos)


def empty_debugger_steps(context):
    return DextIR(
        executable_path=context.options.executable,
        source_paths=context.options.source_files,
        dexter_version=context.version)


def get_debugger_steps(context):
    step_collection = empty_debugger_steps(context)

    with Timer('parsing commands'):
        try:
            step_collection.commands = _get_command_infos(context)
        except CommandParseError as e:
            msg = 'parser error: <d>{}({}):</> {}\n{}\n{}\n'.format(
                e.filename, e.lineno, e.info, e.src, e.caret)
            raise DebuggerException(msg)

    with NamedTemporaryFile(
            dir=context.working_directory.path, delete=False) as fp:
        pickle.dump(step_collection, fp, protocol=pickle.HIGHEST_PROTOCOL)
        steps_path = fp.name

    with NamedTemporaryFile(
            dir=context.working_directory.path, delete=False, mode='wb') as fp:
        pickle.dump(context.options, fp, protocol=pickle.HIGHEST_PROTOCOL)
        options_path = fp.name

    dexter_py = os.path.basename(sys.argv[0])
    if not os.path.isfile(dexter_py):
        dexter_py = os.path.join(get_root_directory(), '..', dexter_py)
    assert os.path.isfile(dexter_py)

    with NamedTemporaryFile(dir=context.working_directory.path) as fp:
        args = [
            sys.executable, dexter_py, 'run-debugger-internal-', steps_path,
            options_path, '--working-directory', context.working_directory.path,
            '--unittest=off', '--indent-timer-level={}'.format(Timer.indent + 2)
        ]
        try:
            with Timer('running external debugger process'):
                subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            raise DebuggerException(e)

    with open(steps_path, 'rb') as fp:
        step_collection = pickle.load(fp)

    return step_collection


class Debuggers(object):
    @classmethod
    def potential_debuggers(cls):
        try:
            return cls._potential_debuggers
        except AttributeError:
            cls._potential_debuggers = _get_potential_debuggers()
            return cls._potential_debuggers

    def __init__(self, context):
        self.context = context

    def load(self, key, step_collection=None):
        with Timer('load {}'.format(key)):
            return Debuggers.potential_debuggers()[key](self.context,
                                                        step_collection)

    def _populate_debugger_cache(self):
        debuggers = []
        for key in sorted(Debuggers.potential_debuggers()):
            debugger = self.load(key)

            class LoadedDebugger(object):
                pass

            LoadedDebugger.option_name = key
            LoadedDebugger.full_name = '[{}]'.format(debugger.name)
            LoadedDebugger.is_available = debugger.is_available

            if LoadedDebugger.is_available:
                try:
                    LoadedDebugger.version = debugger.version.splitlines()
                except AttributeError:
                    LoadedDebugger.version = ['']
            else:
                try:
                    LoadedDebugger.error = debugger.loading_error.splitlines()
                except AttributeError:
                    LoadedDebugger.error = ['']

                try:
                    LoadedDebugger.error_trace = debugger.loading_error_trace
                except AttributeError:
                    LoadedDebugger.error_trace = None

            debuggers.append(LoadedDebugger)
        return debuggers

    def list(self):
        debuggers = self._populate_debugger_cache()

        max_o_len = max(len(d.option_name) for d in debuggers)
        max_n_len = max(len(d.full_name) for d in debuggers)

        msgs = []

        for d in debuggers:
            # Option name, right padded with spaces for alignment
            option_name = (
                '{{name: <{}}}'.format(max_o_len).format(name=d.option_name))

            # Full name, right padded with spaces for alignment
            full_name = ('{{name: <{}}}'.format(max_n_len)
                         .format(name=d.full_name))

            if d.is_available:
                name = '<b>{} {}</>'.format(option_name, full_name)

                # If the debugger is available, show the first line of the
                #  version info.
                available = '<g>YES</>'
                info = '<b>({})</>'.format(d.version[0])
            else:
                name = '<y>{} {}</>'.format(option_name, full_name)

                # If the debugger is not available, show the first line of the
                # error reason.
                available = '<r>NO</> '
                info = '<y>({})</>'.format(d.error[0])

            msg = '{} {} {}'.format(name, available, info)

            if self.context.options.verbose:
                # If verbose mode and there was more version or error output
                # than could be displayed in a single line, display the whole
                # lot slightly indented.
                verbose_info = None
                if d.is_available:
                    if d.version[1:]:
                        verbose_info = d.version + ['\n']
                else:
                    # Some of list elems may contain multiple lines, so make
                    # sure each elem is a line of its own.
                    verbose_info = d.error_trace

                if verbose_info:
                    verbose_info = '\n'.join('        {}'.format(l.rstrip())
                                             for l in verbose_info) + '\n'
                    msg = '{}\n\n{}'.format(msg, verbose_info)

            msgs.append(msg)
        self.context.o.auto('\n{}\n\n'.format('\n'.join(msgs)))
