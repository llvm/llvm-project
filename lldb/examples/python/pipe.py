r"""
Pipe or redirect LLDB command output through shell commands.

Usage:
    (lldb) pipe <lldb-command> | <shell-pipeline>
    (lldb) pipe <lldb-command> > <file>
Examples:
    (lldb) pipe image list | wc -l
    (lldb) pipe settings list | grep color
    (lldb) pipe bt all | grep Foundation
    (lldb) pipe bt all > /tmp/stack.txt
    (lldb) pipe bt all | tee /tmp/stack.txt
    (lldb) pipe frame variable *this | pbcopy
    (lldb) pipe breakpoint list | grep 'where = .*resolved'
    (lldb) pipe register read | sort

Import:
    (lldb) command script import lldb.utils.pipe
"""

import os
import shlex
import subprocess
from typing import Iterable, Iterator, Optional, Tuple

import lldb


# Equivalent to itertools.pairwise, available in Python 3.10+.
def _pairwise(iterable: Iterable) -> Iterator[Tuple]:
    """Yield consecutive overlapping pairs: _pairwise(ABCD) -> AB BC CD."""
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def _tokenize(cmdstr: str) -> shlex.shlex:
    """Splits on whitespace and treats | and > as standalone punctuation, even
    without surrounding spaces."""

    lex = shlex.shlex(cmdstr, posix=True, punctuation_chars="|>")
    lex.whitespace_split = True
    return lex


def _split_cmd(cmdstr: str) -> Tuple[str, Optional[str]]:
    """Split cmdstr at the first unquoted shell operator (| or >)."""
    lex = _tokenize(cmdstr)
    for token in lex:
        if token in ("|", ">"):
            # Search up to where shlex has consumed.
            split = cmdstr.rindex(token, 0, lex.instream.tell())
            first_cmd = cmdstr[:split].rstrip()
            shell_suffix = cmdstr[split:]
            return first_cmd, shell_suffix
    return cmdstr, None


def _pager_names() -> set[str]:
    """Return known pager command names, including $PAGER if set."""
    pagers = {"less", "more", "most", "bat"}
    if pager := os.environ.get("PAGER"):
        # shlex.split handles flags, e.g. "less -R" -> "less".
        pager = shlex.split(pager)[0]
        pagers.add(os.path.basename(pager))
    return pagers


_PAGERS = _pager_names()


def _ends_with_pager(shell_cmd: str) -> bool:
    """Check if the last command in a shell pipeline is a pager."""
    lex = _tokenize(shell_cmd)
    for tok1, tok2 in reversed(list(_pairwise(lex))):
        if tok1 == "|":
            return os.path.basename(tok2) in _PAGERS
    return False


def _is_lldb_command(interp: lldb.SBCommandInterpreter, cmdstr: str) -> bool:
    result = lldb.SBCommandReturnObject()
    interp.ResolveCommand(cmdstr, result)
    return result.Succeeded()


@lldb.command()
def pipe(
    debugger: lldb.SBDebugger,
    cmdstr: str,
    exe_ctx: lldb.SBExecutionContext,
    result: lldb.SBCommandReturnObject,
    _,
):
    """Pipe or redirect LLDB command output through shell commands."""
    interp = debugger.GetCommandInterpreter()

    first_cmd, shell_suffix = _split_cmd(cmdstr)

    if not first_cmd and shell_suffix:
        result.SetError("no command before shell operator")
        return

    if _is_lldb_command(interp, first_cmd):
        if shell_suffix is None:
            # Run a single lldb command, shell command to pipe to.
            interp.HandleCommand(first_cmd, exe_ctx, result)
            return

        cmd_result = lldb.SBCommandReturnObject()
        interp.HandleCommand(first_cmd, exe_ctx, cmd_result)

        if cmd_result.GetError():
            result.SetError(cmd_result.GetError())
        if not cmd_result.Succeeded():
            if not cmd_result.GetError():
                result.SetError(f"command failed: {first_cmd}")
            return

        output = cmd_result.GetOutput()
        # `cat` works with both | and > operators.
        shell_cmd = f"cat {shell_suffix}"
    else:
        # No lldb command to run, only shell.
        output = None
        shell_cmd = cmdstr

    try:
        if _ends_with_pager(shell_cmd):
            proc = subprocess.Popen(shell_cmd, shell=True, stdin=subprocess.PIPE)
            proc.communicate(output.encode() if output else None)
        else:
            proc = subprocess.run(
                shell_cmd, shell=True, input=output, capture_output=True, text=True
            )
            if proc.stdout:
                result.PutCString(proc.stdout)
            if proc.stderr:
                result.SetError(proc.stderr)
    except OSError as e:
        result.SetError(f"failed to run shell command: {e}")
        return
