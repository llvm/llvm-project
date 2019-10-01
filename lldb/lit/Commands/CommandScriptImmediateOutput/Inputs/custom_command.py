from __future__ import print_function

import sys


def split(command):
    command = command.strip()
    return command.rsplit(' ', 1)

def command_function(debugger, command, exe_ctx, result, internal_dict):
    result.SetImmediateOutputFile(sys.__stdout__)
    print('this is a test string, just a test string', file=result)


def write_file(debugger, command, exe_ctx, result, internal_dict):
    args = split(command)
    path = args[0]
    mode = args[1]
    with open(path, mode) as f:
        result.SetImmediateOutputFile(f)
        if not mode in ['r']:
            print('writing to file with mode: ' + mode, file=result)
