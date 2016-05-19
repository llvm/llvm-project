from __future__ import print_function

import sys

def command_function(debugger, command, exe_ctx, result, internal_dict):
        result.SetImmediateOutputFile(sys.__stdout__)
        print('this is a test string, just a test string', file=result)

