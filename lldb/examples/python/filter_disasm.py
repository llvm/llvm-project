"""
Defines a command, fdis, that does filtered disassembly. The command does the
lldb disassemble command with -b and any other arguments passed in, and
pipes that through a provided filter program.

The intention is to support disassembly of RISC-V proprietary instructions.
This is handled with llvm-objdump by piping the output of llvm-objdump through
a filter program. This script is intended to mimic that workflow.
"""

import lldb
import subprocess


class Program(list):
    def __str__(self):
        return " ".join(self)


filter_program = Program(["crustfilt"])

def __lldb_init_module(debugger, dict):
    debugger.HandleCommand("command script add -f filter_disasm.fdis fdis")
    print("Disassembly filter command (fdis) loaded")
    print("Filter program set to %s" % filter_program)


def fdis(debugger, args, exe_ctx, result, dict):
    """
  Call the built in disassembler, then pass its output to a filter program
  to add in disassembly for hidden opcodes.
  Except for get and set, use the fdis command like the disassemble command.
  By default, the filter program is crustfilt, from
  https://github.com/quic/crustfilt . This can be changed by changing
  the global variable filter_program.

  Usage:
    fdis [[get] [set <program>] [<disassembly options>]]

    Choose one of the following:
        get
            Gets the current filter program

        set <program>
            Sets the current filter program. This can be an executable, which
            will be found on PATH, or an absolute path.

        <disassembly options>
            If the first argument is not get or set, the args will be passed
            to the disassemble command as is.

    """

    global filter_program
    args_list = args.split(" ")
    result.Clear()

    if len(args_list) == 1 and args_list[0] == "get":
        result.PutCString(str(filter_program))
        result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
        return

    if args_list[0] == "set":
        # Assume the rest is a program to run and any arguments to be passed to
        # it.
        if len(args_list) <= 1:
            result.PutCString('"set" command requires a program argument')
            result.SetStatus(lldb.eReturnStatusFailed)
            return

        filter_program = Program(args_list[1:])
        result.PutCString('Filter program set to "{}"'.format(filter_program))
        result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
        return

    res = lldb.SBCommandReturnObject()
    debugger.GetCommandInterpreter().HandleCommand("disassemble -b " + args, exe_ctx, res)
    if len(res.GetError()) > 0:
        result.SetError(res.GetError())
        result.SetStatus(lldb.eReturnStatusFailed)
        return
    output = res.GetOutput()

    try:
        proc = subprocess.run(
            filter_program, capture_output=True, text=True, input=output
        )
    except (subprocess.SubprocessError, OSError) as e:
        result.PutCString("Error occurred. Original disassembly:\n\n" + output)
        result.SetError(str(e))
        result.SetStatus(lldb.eReturnStatusFailed)
        return

    if proc.returncode:
        result.PutCString("warning: {} returned non-zero value {}".format(filter_program, proc.returncode))

    result.PutCString(proc.stdout)
    result.SetStatus(lldb.eReturnStatusSuccessFinishResult)
