# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Provides Dexter-specific exception types."""


class Dexception(Exception):
    """All dexter-specific exceptions derive from this."""
    pass


class Error(Dexception):
    """Error.  Prints 'error: <message>' without a traceback."""
    pass


class DebuggerException(Dexception):
    """Any error from using the debugger."""

    def __init__(self, msg, orig_exception=None):
        super(DebuggerException, self).__init__(msg)
        self.msg = msg
        self.orig_exception = orig_exception

    def __str__(self):
        return str(self.msg)


class LoadDebuggerException(DebuggerException):
    """If specified debugger cannot be loaded."""
    pass


class NotYetLoadedDebuggerException(LoadDebuggerException):
    """If specified debugger has not yet been attempted to load."""

    def __init__(self):
        super(NotYetLoadedDebuggerException,
              self).__init__('not loaded', orig_exception=None)


class CommandParseError(Dexception):
    """If a command instruction cannot be successfully parsed."""

    def __init__(self, *args, **kwargs):
        super(CommandParseError, self).__init__(*args, **kwargs)
        self.filename = None
        self.lineno = None
        self.info = None
        self.src = None
        self.caret = None


class NonFloatValueInCommand(CommandParseError):
    """If a command has the float_range arg but at least one of its expected
    values cannot be converted to a float."""

    def __init__(self, *args, **kwargs):
        super(NonFloatValueInCommand, self).__init__(*args, **kwargs)
        self.value = None

class ToolArgumentError(Dexception):
    """If a tool argument is invalid."""
    pass


class BuildScriptException(Dexception):
    """If there is an error in a build script file."""

    def __init__(self, *args, **kwargs):
        self.script_error = kwargs.pop('script_error', None)
        super(BuildScriptException, self).__init__(*args, **kwargs)


class HeuristicException(Dexception):
    """If there was a problem with the heuristic."""
    pass
