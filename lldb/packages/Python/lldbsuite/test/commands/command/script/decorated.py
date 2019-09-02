from __future__ import print_function

import lldb


@lldb.command()
def decorated1(debugger, args, exe_ctx, result, dict):
    """
    Python command defined by @lldb.command
    """
    print("hello from decorated1", file=result)


@lldb.command(doc="Python command defined by @lldb.command")
def decorated2(debugger, args, exe_ctx, result, dict):
    """
    This docstring is overridden.
    """
    print("hello from decorated2", file=result)


@lldb.command()
def decorated3(debugger, args, result, dict):
    """
    Python command defined by @lldb.command
    """
    print("hello from decorated3", file=result)


@lldb.command("decorated4")
def _decorated4(debugger, args, exe_ctx, result, dict):
    """
    Python command defined by @lldb.command
    """
    print("hello from decorated4", file=result)
