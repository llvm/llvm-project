#!/usr/bin/env python3

# ----------------------------------------------------------------------
# Be sure to add the python path that points to the LLDB shared library.
# On MacOSX csh, tcsh:
#   setenv PYTHONPATH /Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
# On MacOSX sh, bash:
#   export PYTHONPATH=/Developer/Library/PrivateFrameworks/LLDB.framework/Resources/Python
# On debian
#   export PYTHONPATH=`lldb -P`
#   export LLDB_DEBUGSERVER_PATH=/usr/bin/lldb-server
# ----------------------------------------------------------------------

import lldb
import os
import sys


def usage():
    print("Usage: diffdbg.py [-n name] executable")
    print("Run two copies of `executable` until a breakpoint at function")
    print("`name` (default 'main') then step until execution differs.")
    print("Manually edit `argsA` and `argsB` to pass different arguments.")
    sys.exit(0)


def print_frames(thread):
    for frame in thread.frames:
        print(frame)


if len(sys.argv) == 2:
    fname = "main"
    exe = sys.argv[1]
elif len(sys.argv) == 4:
    if sys.argv[1] != "-n":
        usage()
    else:
        fname = sys.argv[2]
        exe = sys.argv[3]
else:
    usage()

argsA = []
argsB = []

# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process
# stops. We do this by setting the async mode to false.
debugger.SetAsync(False)

# Create a target from a file and arch
print("Creating targets for '%s'" % exe)

targetA = debugger.CreateTargetWithFileAndArch(exe, lldb.LLDB_ARCH_DEFAULT)
targetB = debugger.CreateTargetWithFileAndArch(exe, lldb.LLDB_ARCH_DEFAULT)

if targetA and targetB:
    # If the target is valid set a breakpoint at main
    fileA = targetA.GetExecutable().GetFilename()
    fileB = targetB.GetExecutable().GetFilename()
    main_bpA = targetA.BreakpointCreateByName(fname, fileA)
    main_bpB = targetB.BreakpointCreateByName(fname, fileB)

    print(main_bpA)
    print(main_bpB)

    # Launch the process. Since we specified synchronous mode, we won't return
    # from this function until we hit the breakpoint at main
    processA = targetA.LaunchSimple(argsA, None, os.getcwd())
    processB = targetB.LaunchSimple(argsB, None, os.getcwd())

    # Make sure the launch went ok
    if processA and processB:
        # Print some simple process info
        print(processA)
        print(processB)
        stateA = processA.GetState()
        stateB = processB.GetState()
        if stateA == lldb.eStateStopped and stateB == lldb.eStateStopped:
            # Get the first thread
            threadA = processA.GetThreadAtIndex(0)
            threadB = processB.GetThreadAtIndex(0)
            if threadA and threadB:
                # Print some simple thread info
                print(threadA)
                print(threadB)

                done = False
                while not done:
                    errorA = lldb.SBError()
                    errorB = lldb.SBError()
                    threadA.StepInstruction(False, errorA)
                    threadB.StepInstruction(False, errorB)
                    if not errorA.Success():
                        print(errorA.description)
                        done = True
                        continue
                    if not errorB.Success():
                        print(errorB.description)
                        done = True
                        continue

                    # Get the first frame
                    frameA = threadA.GetFrameAtIndex(0)
                    frameB = threadB.GetFrameAtIndex(0)
                    if frameA and frameB:
                        addrA = frameA.addr
                        addrB = frameB.addr
                        if addrA and addrB:
                            if addrA != addrB:
                                print_frames(threadA)
                                print_frames(threadB)
                                done = True
                                continue

            print("execution diverged, press enter to continue and wait for")
            print("program to exit or 'Ctrl-D'/'quit' to terminate the program")
            next = sys.stdin.readline()
            if not next or next.rstrip("\n") == "quit":
                print("Terminating the inferior process...")
                processA.Kill()
                processB.Kill()
            else:
                # Now continue to the program exit
                processA.Continue()
                processB.Continue()
                # When we return from the above function we will hopefully be at the
                # program exit. Print out some process info
                print(processA)
                print(processB)
        elif state == lldb.eStateExited:
            print("Didn't hit the breakpoint at main, program has exited...")
        else:
            print(
                "Unexpected process state: %s, killing process..."
                % debugger.StateAsCString(state)
            )
            processA.Kill()
            processB.Kill()


lldb.SBDebugger.Terminate()
