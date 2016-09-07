"""
                     The LLVM Compiler Infrastructure

 This file is distributed under the University of Illinois Open Source
 License. See LICENSE.TXT for details.

Configuration options for lldbtest.py set by dotest.py during initialization
"""

# array of strings
# each string has the name of an lldb channel followed by
# zero or more categories in that channel
# ex. "gdb-remote packets"
channels = []

# leave logs/traces even for successful test runs
log_success = False

# path to the lldb command line executable tool
lldbExec = None
