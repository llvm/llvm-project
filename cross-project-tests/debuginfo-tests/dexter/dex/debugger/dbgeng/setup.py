# DExTer : Debugging Experience Tester
# ~~~~~~   ~         ~~         ~   ~~
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ctypes import *

from . import client
from . import control
from . import symbols
from .probe_process import probe_state
from .utils import *


class STARTUPINFOA(Structure):
    _fields_ = [
        ("cb", c_ulong),
        ("lpReserved", c_char_p),
        ("lpDesktop", c_char_p),
        ("lpTitle", c_char_p),
        ("dwX", c_ulong),
        ("dwY", c_ulong),
        ("dwXSize", c_ulong),
        ("dwYSize", c_ulong),
        ("dwXCountChars", c_ulong),
        ("dwYCountChars", c_ulong),
        ("dwFillAttribute", c_ulong),
        ("wShowWindow", c_ushort),
        ("cbReserved2", c_ushort),
        ("lpReserved2", c_char_p),
        ("hStdInput", c_void_p),
        ("hStdOutput", c_void_p),
        ("hStdError", c_void_p),
    ]


class PROCESS_INFORMATION(Structure):
    _fields_ = [
        ("hProcess", c_void_p),
        ("hThread", c_void_p),
        ("dwProcessId", c_ulong),
        ("dwThreadId", c_ulong),
    ]


def fetch_local_function_syms(Symbols, prefix):
    syms = Symbols.get_all_functions()

    def is_sym_in_src_dir(sym):
        name, data = sym
        symdata = Symbols.GetLineByOffset(data.Offset)
        if symdata is not None:
            srcfile, line = symdata
            if prefix in srcfile:
                return True
        return False

    syms = [x for x in syms if is_sym_in_src_dir(x)]
    return syms


def break_on_all_but_main(Control, Symbols, main_offset):
    mainfile, _ = Symbols.GetLineByOffset(main_offset)
    prefix = "\\".join(mainfile.split("\\")[:-1])

    for name, rec in fetch_local_function_syms(Symbols, prefix):
        if name == "main":
            continue
        bp = Control.AddBreakpoint2(offset=rec.Offset, enabled=True)

    # All breakpoints are currently discarded: we just sys.exit for cleanup
    return


def setup_everything(binfile):
    from . import client
    from . import symbols

    Client = client.Client()

    Client.Control.SetEngineOptions(0x20)  # DEBUG_ENGOPT_INITIAL_BREAK

    Client.CreateProcessAndAttach2(binfile)

    # Load lines as well as general symbols
    sym_opts = Client.Symbols.GetSymbolOptions()
    sym_opts |= symbols.SymbolOptionFlags.SYMOPT_LOAD_LINES
    Client.Symbols.SetSymbolOptions(sym_opts)

    # Need to enter the debugger engine to let it attach properly.
    res = Client.Control.WaitForEvent(timeout=1000)
    if res == S_FALSE:
        # The debugee apparently didn't do anything at all. Rather than risk
        # hanging, bail out at this point.
        client.TerminateProcesses()
        raise Exception("Debuggee did not start in a timely manner")

    # Enable line stepping.
    Client.Control.Execute("l+t")
    # Enable C++ expression interpretation.
    Client.Control.SetExpressionSyntax(cpp=True)

    # We've requested to break into the process at the earliest opportunity,
    # and WaitForEvent'ing means we should have reached that break state.
    # Now set a breakpoint on the main symbol, and "go" until we reach it.
    module_name = Client.Symbols.get_exefile_module_name()
    offset = Client.Symbols.GetOffsetByName("{}!main".format(module_name))
    breakpoint = Client.Control.AddBreakpoint2(offset=offset, enabled=True)
    Client.Control.SetExecutionStatus(control.DebugStatus.DEBUG_STATUS_GO)

    # Problem: there is no guarantee that the client will ever reach main,
    # something else exciting could happen in that time, the host system may
    # be very loaded, and similar. Wait for some period, say, five seconds, and
    # abort afterwards: this is a trade-off between spurious timeouts and
    # completely hanging in the case of a environmental/programming error.
    res = Client.Control.WaitForEvent(timeout=5000)
    if res == S_FALSE:
        client.TerminateProcesses()
        raise Exception("Debuggee did not reach main function in a timely manner")

    break_on_all_but_main(Client.Control, Client.Symbols, offset)

    # Set the default action on all exceptions to be "quit and detach". If we
    # don't, dbgeng will merrily spin at the exception site forever.
    filts = Client.Control.GetNumberEventFilters()
    for x in range(filts[0], filts[0] + filts[1]):
        Client.Control.SetExceptionFilterSecondCommand(x, "qd")

    return Client


def step_once(client):
    client.Control.Execute("p")
    try:
        client.Control.WaitForEvent()
    except Exception as e:
        if (
            client.Control.GetExecutionStatus()
            == control.DebugStatus.DEBUG_STATUS_NO_DEBUGGEE
        ):
            return None  # Debuggee has gone away, likely due to an exception.
        raise e
    # Could assert here that we're in the "break" state
    client.Control.GetExecutionStatus()
    return probe_state(client)


def main_loop(client):
    res = True
    while res is not None:
        res = step_once(client)


def cleanup(client):
    client.TerminateProcesses()
