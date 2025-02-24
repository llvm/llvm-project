import os
import re
import sys
import subprocess

import lldb


@lldb.command()
def fzf_history(debugger, cmdstr, ctx, result, _):
    """Use fzf to search and select from lldb command history."""
    history_file = os.path.expanduser("~/.lldb/lldb-widehistory")
    if not os.path.exists(history_file):
        result.SetError("history file does not exist")
        return
    history = _load_history(history_file)

    if sys.platform != "darwin":
        # The ability to integrate fzf's result into lldb uses copy and paste.
        # In absense of copy and paste, do the basics and forgo features.
        fzf_command = ("fzf", "--no-sort", f"--query={query}")
        subprocess.run(fzf_command, input=history)
        return

    # Capture the current pasteboard contents to restore after overwriting.
    paste_snapshot = subprocess.run("pbpaste", text=True, capture_output=True).stdout

    # On enter, copy the selected history entry into the pasteboard.
    fzf_command = (
        "fzf",
        "--no-sort",
        f"--query={cmdstr}",
        "--bind=enter:execute-silent(echo -n {} | pbcopy)+close",
    )
    completed = subprocess.run(fzf_command, input=history, text=True)
    # 130 is used for CTRL-C or ESC.
    if completed.returncode not in (0, 130):
        result.SetError("fzf failed")
        return

    # Get the user's selected history entry.
    selected_command = subprocess.run("pbpaste", text=True, capture_output=True).stdout
    if selected_command == paste_snapshot:
        # Nothing was selected, no cleanup needed.
        return

    _handle_command(debugger, selected_command)

    # Restore the pasteboard's contents.
    subprocess.run("pbcopy", input=paste_snapshot, text=True)


def _handle_command(debugger, command):
    """Try pasting the command, and failing that, run it directly."""
    if not command:
        return

    # Use applescript to paste the selected result into lldb's console.
    paste_command = (
        "osascript",
        "-e",
        'tell application "System Events" to keystroke "v" using command down',
    )
    completed = subprocess.run(paste_command, capture_output=True)

    if completed.returncode != 0:
        # The above applescript requires the "control your computer" permission.
        #     Settings > Private & Security > Accessibility
        # If not enabled, fallback to running the command.
        debugger.HandleCommand(command)


def _load_history(history_file):
    """Load, decode, parse, and prepare an lldb history file for fzf."""
    with open(history_file) as f:
        history_contents = f.read()

    history_decoded = re.sub(r"\\0([0-7][0-7])", _decode_char, history_contents)
    history_lines = history_decoded.splitlines()

    # Skip the header line (_HiStOrY_V2_)
    del history_lines[0]
    # Reverse to show latest first.
    history_lines.reverse()

    history_commands = []
    history_seen = set()
    for line in history_lines:
        line = line.strip()
        # Skip empty lines, single character commands, and duplicates.
        if line and len(line) > 1 and line not in history_seen:
            history_commands.append(line)
            history_seen.add(line)

    return "\n".join(history_commands)


def _decode_char(match):
    """Decode octal strings ('\0NN') into a single character string."""
    code = int(match.group(1), base=8)
    return chr(code)
