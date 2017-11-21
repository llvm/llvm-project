#!/bin/env python
"""
A gdb-compatible frontend for lldb that implements just enough
commands to run the tests in the debuginfo-tests repository with lldb.
"""

# ----------------------------------------------------------------------
# Auto-detect lldb python module.
import commands, platform, os,  sys
try:
    # Just try for LLDB in case PYTHONPATH is already correctly setup.
    import lldb
except ImportError:
    lldb_python_dirs = list()
    # lldb is not in the PYTHONPATH, try some defaults for the current platform.
    platform_system = platform.system()
    if platform_system == 'Darwin':
        # On Darwin, try the currently selected Xcode directory
        xcode_dir = commands.getoutput("xcode-select --print-path")
        if xcode_dir:
            lldb_python_dirs.append(os.path.realpath(xcode_dir +
'/../SharedFrameworks/LLDB.framework/Resources/Python'))
            lldb_python_dirs.append(xcode_dir +
'/Library/PrivateFrameworks/LLDB.framework/Resources/Python')
        lldb_python_dirs.append(
'/System/Library/PrivateFrameworks/LLDB.framework/Resources/Python')
    success = False
    for lldb_python_dir in lldb_python_dirs:
        if os.path.exists(lldb_python_dir):
            if not (sys.path.__contains__(lldb_python_dir)):
                sys.path.append(lldb_python_dir)
                try:
                    import lldb
                except ImportError:
                    pass
                else:
                    print 'imported lldb from: "%s"' % (lldb_python_dir)
                    success = True
                    break
    if not success:
        print "error: couldn't locate the 'lldb' module, please set PYTHONPATH correctly"
        sys.exit(1)
# ----------------------------------------------------------------------

# Command line option handling.
import argparse
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--quiet', '-q', action="store_true", help='ignored')
parser.add_argument('-batch', action="store_true",
                    help='exit after processing comand line')
parser.add_argument('-n', action="store_true", help='ignore .lldb file')
parser.add_argument('-x', dest='script', type=file, help='execute commands from file')
parser.add_argument("target", help="the program to debug")
args = parser.parse_args()


# Create a new debugger instance.
debugger = lldb.SBDebugger.Create()
debugger.SkipLLDBInitFiles(args.n)

# Don't return from lldb function calls until the process stops.
debugger.SetAsync(False)

# Create a target from a file and arch.
arch = os.popen("file "+args.target).read().split()[-1]
target = debugger.CreateTargetWithFileAndArch(args.target, arch)

if not target:
    print "Could not create target", args.target
    sys.exit(1)

if not args.script:
    print "Interactive mode is not implemented."
    sys.exit(1)

import re
for command in args.script:
    # Strip newline and whitespaces and split into words.
    cmd = command[:-1].strip().split()
    if not cmd:
        continue

    print '> %s'% command[:-1]

    try:
        if re.match('^r|(run)$', cmd[0]):
            error = lldb.SBError()
            launchinfo = lldb.SBLaunchInfo([])
            launchinfo.SetWorkingDirectory(os.getcwd())
            process = target.Launch(launchinfo, error)
            print error
            if not process or error.fail:
                state = process.GetState()
                print "State = %d" % state
                print """
ERROR: Could not launch process.
NOTE: There are several reasons why this may happen:
  * Root needs to run "DevToolsSecurity --enable".
  * Older versions of lldb cannot launch more than one process simultaneously.
"""
                sys.exit(1)

        elif re.match('^b|(break)$', cmd[0]) and len(cmd) == 2:
            if re.match('[0-9]+', cmd[1]):
                # b line
                mainfile = target.FindFunctions('main')[0].compile_unit.file
                print target.BreakpointCreateByLocation(mainfile, int(cmd[1]))
            else:
                # b file:line
                file, line = cmd[1].split(':')
                print target.BreakpointCreateByLocation(file, int(line))

        elif re.match('^ptype$', cmd[0]) and len(cmd) == 2:
            # GDB's ptype has multiple incarnations depending on its
            # argument (global variable, function, type).  The definition
            # here is for looking up the signature of a function and only
            # if that fails it looks for a type with that name.
            # Type lookup in LLDB would be "image lookup --type".
            for elem in target.FindFunctions(cmd[1]):
                print elem.function.type
                continue
            print target.FindFirstType(cmd[1])

        elif re.match('^po$', cmd[0]) and len(cmd) > 1:
            try:
                opts = lldb.SBExpressionOptions()
                opts.SetFetchDynamicValue(True)
                opts.SetCoerceResultToId(True)
                print target.EvaluateExpression(' '.join(cmd[1:]), opts)
            except:
                # FIXME: This is a fallback path for the lab.llvm.org
                # buildbot running OS X 10.7; it should be removed.
                thread = process.GetThreadAtIndex(0)
                frame = thread.GetFrameAtIndex(0)
                print frame.EvaluateExpression(' '.join(cmd[1:]))

        elif re.match('^p|(print)$', cmd[0]) and len(cmd) > 1:
            thread = process.GetThreadAtIndex(0)
            frame = thread.GetFrameAtIndex(0)
            print frame.EvaluateExpression(' '.join(cmd[1:]))

        elif re.match('^n|(next)$', cmd[0]):
            thread = process.GetThreadAtIndex(0)
            thread.StepOver()

        elif re.match('^q|(quit)$', cmd[0]):
            sys.exit(0)

        else:
            print debugger.HandleCommand(' '.join(cmd))

    except SystemExit:
        lldb.SBDebugger_Terminate()
        raise
    except:
        print 'Could not handle the command "%s"' % ' '.join(cmd)

