#!/usr/bin/env python3

# Kills all processes whos command line match provided pattern

import os
import signal
import subprocess
import sys
import platform


def windows_impl(search_pattern):
    cmd = "wmic process get ProcessId,CommandLine"
    output = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    lines = output.split("\n")

    PIDs = []

    for line in lines:
        if "clang" in line and "kill_process" not in line and "llvm-lit" not in line:
            # creates something like ['..\\llvm-project\\build\\bin\\clang.exe  -cc1modbuildd mbd-launch -v', '16260']
            parts = line.rsplit(None, 1)

            if len(parts) == 2:
                command_line, pid = parts
                if search_pattern in command_line:
                    PIDs.append(int(pid))

    return PIDs


def unix_impl(search_pattern):
    ps_output = subprocess.check_output(["ps", "aux"], universal_newlines=True)
    ps_lines = ps_output.split("\n")

    PIDs = []

    for line in ps_lines:
        if "clang" in line and "kill_process" not in line and "llvm-lit" not in line:
            ps_line_components = line.split()

            # output of ps aux
            # 0    1   2    3    4   5   6   7    8     9    10
            # USER PID %CPU %MEM VSZ RSS TTY STAT START TIME COMMAND

            # skip line without COMMAND
            if len(ps_line_components) < 11:
                continue

            command_line = " ".join(ps_line_components[10:])
            if search_pattern in command_line:
                PIDs.append(int(ps_line_components[1]))

    return PIDs


def main():
    if len(sys.argv) == 1:
        sys.stderr.write("Error: no search pattern provided\n")
        sys.exit(1)

    search_pattern = sys.argv[1]
    sys.stdout.write(
        f"Searching for process with pattern '{search_pattern}' in command line\n"
    )

    system_name = platform.system()

    PIDs = []
    if system_name == "Windows":
        PIDs = windows_impl(search_pattern)
    else:
        PIDs = unix_impl(search_pattern)

    if len(PIDs) == 0:
        sys.stdout.write("No processes matching search pattern were found\n")
        return

    sys.stdout.write(f"Killing PIDs {PIDs}\n")
    for PID in PIDs:
        os.kill(PID, signal.SIGTERM)


if __name__ == "__main__":
    main()
