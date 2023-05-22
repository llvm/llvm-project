%feature("docstring",
"SBDebugger is the primordial object that creates SBTargets and provides
access to them.  It also manages the overall debugging experiences.

For example (from example/disasm.py),::

    import lldb
    import os
    import sys

    def disassemble_instructions (insts):
        for i in insts:
            print i

    ...

    # Create a new debugger instance
    debugger = lldb.SBDebugger.Create()

    # When we step or continue, don't return from the function until the process
    # stops. We do this by setting the async mode to false.
    debugger.SetAsync (False)

    # Create a target from a file and arch
    print('Creating a target for \'%s\'' % exe)

    target = debugger.CreateTargetWithFileAndArch (exe, lldb.LLDB_ARCH_DEFAULT)

    if target:
        # If the target is valid set a breakpoint at main
        main_bp = target.BreakpointCreateByName (fname, target.GetExecutable().GetFilename());

        print main_bp

        # Launch the process. Since we specified synchronous mode, we won't return
        # from this function until we hit the breakpoint at main
        process = target.LaunchSimple (None, None, os.getcwd())

        # Make sure the launch went ok
        if process:
            # Print some simple process info
            state = process.GetState ()
            print process
            if state == lldb.eStateStopped:
                # Get the first thread
                thread = process.GetThreadAtIndex (0)
                if thread:
                    # Print some simple thread info
                    print thread
                    # Get the first frame
                    frame = thread.GetFrameAtIndex (0)
                    if frame:
                        # Print some simple frame info
                        print frame
                        function = frame.GetFunction()
                        # See if we have debug info (a function)
                        if function:
                            # We do have a function, print some info for the function
                            print function
                            # Now get all instructions for this function and print them
                            insts = function.GetInstructions(target)
                            disassemble_instructions (insts)
                        else:
                            # See if we have a symbol in the symbol table for where we stopped
                            symbol = frame.GetSymbol();
                            if symbol:
                                # We do have a symbol, print some info for the symbol
                                print symbol
                                # Now get all instructions for this symbol and print them
                                insts = symbol.GetInstructions(target)
                                disassemble_instructions (insts)

                        registerList = frame.GetRegisters()
                        print('Frame registers (size of register set = %d):' % registerList.GetSize())
                        for value in registerList:
                            #print value
                            print('%s (number of children = %d):' % (value.GetName(), value.GetNumChildren()))
                            for child in value:
                                print('Name: ', child.GetName(), ' Value: ', child.GetValue())

                print('Hit the breakpoint at main, enter to continue and wait for program to exit or \'Ctrl-D\'/\'quit\' to terminate the program')
                next = sys.stdin.readline()
                if not next or next.rstrip('\\n') == 'quit':
                    print('Terminating the inferior process...')
                    process.Kill()
                else:
                    # Now continue to the program exit
                    process.Continue()
                    # When we return from the above function we will hopefully be at the
                    # program exit. Print out some process info
                    print process
            elif state == lldb.eStateExited:
                print('Didn\'t hit the breakpoint at main, program has exited...')
            else:
                print('Unexpected process state: %s, killing process...' % debugger.StateAsCString (state))
                process.Kill()

Sometimes you need to create an empty target that will get filled in later.  The most common use for this
is to attach to a process by name or pid where you don't know the executable up front.  The most convenient way
to do this is: ::

    target = debugger.CreateTarget('')
    error = lldb.SBError()
    process = target.AttachToProcessWithName(debugger.GetListener(), 'PROCESS_NAME', False, error)

or the equivalent arguments for :py:class:`SBTarget.AttachToProcessWithID` ."
) lldb::SBDebugger;

%feature("docstring",
    "The dummy target holds breakpoints and breakpoint names that will prime newly created targets."
) lldb::SBDebugger::GetDummyTarget;

%feature("docstring",
    "Return true if target is deleted from the target list of the debugger."
) lldb::SBDebugger::DeleteTarget;

%feature("docstring",
    "Get the number of currently active platforms."
) lldb::SBDebugger::GetNumPlatforms;

%feature("docstring",
    "Get one of the currently active platforms."
) lldb::SBDebugger::GetPlatformAtIndex;

%feature("docstring",
    "Get the number of available platforms."
) lldb::SBDebugger::GetNumAvailablePlatforms;

%feature("docstring", "
    Get the name and description of one of the available platforms.

    @param idx Zero-based index of the platform for which info should be
               retrieved, must be less than the value returned by
               GetNumAvailablePlatforms()."
) lldb::SBDebugger::GetAvailablePlatformInfoAtIndex;

%feature("docstring",
"Launch a command interpreter session. Commands are read from standard input or
from the input handle specified for the debugger object. Output/errors are
similarly redirected to standard output/error or the configured handles.

@param[in] auto_handle_events If true, automatically handle resulting events.
@param[in] spawn_thread If true, start a new thread for IO handling.
@param[in] options Parameter collection of type SBCommandInterpreterRunOptions.
@param[in] num_errors Initial error counter.
@param[in] quit_requested Initial quit request flag.
@param[in] stopped_for_crash Initial crash flag.

@return
A tuple with the number of errors encountered by the interpreter, a boolean
indicating whether quitting the interpreter was requested and another boolean
set to True in case of a crash.

Example: ::

    # Start an interactive lldb session from a script (with a valid debugger object
    # created beforehand):
    n_errors, quit_requested, has_crashed = debugger.RunCommandInterpreter(True,
        False, lldb.SBCommandInterpreterRunOptions(), 0, False, False)"
) lldb::SBDebugger::RunCommandInterpreter;
