# Implementing Standalone Scripts

### Configuring `PYTHONPATH`

LLDB has all of its core code built into a shared library which gets used by
the `lldb` command line application.
- On macOS this shared library is a framework: `LLDB.framework`.
- On other unix variants the program is a shared library: lldb.so.

LLDB also provides an `lldb.py` module that contains the bindings from LLDB
into Python. To use the `LLDB.framework` to create your own stand-alone python
programs, you will need to tell python where to look in order to find this
module. This is done by setting the `PYTHONPATH` environment variable,
adding a path to the directory that contains the `lldb.py` python
module. The lldb driver program has an option to report the path to the lldb
module. You can use that to point to correct lldb.py:

For csh and tcsh:

```csh
% setenv PYTHONPATH `lldb -P`
```

For sh and bash:

```bash
$ export PYTHONPATH=`lldb -P`
```

Alternatively, you can append the LLDB Python directory to the sys.path list
directly in your Python code before importing the lldb module.

### Initialization

The standard test for `__main__`, like many python modules do, is useful for
creating scripts that can be run from the command line. However, for command
line scripts, the debugger instance must be created manually. Sample code would
look like:

```python3
if __name__ == '__main__':
    # Initialize the debugger before making any API calls.
    lldb.SBDebugger.Initialize()
    # Create a new debugger instance in your module if your module
    # can be run from the command line. When we run a script from
    # the command line, we won't have any debugger object in
    # lldb.debugger, so we can just create it if it will be needed
    debugger = lldb.SBDebugger.Create()

    # Next, do whatever work this module should do when run as a command.
    # ...

    # Finally, dispose of the debugger you just made.
    lldb.SBDebugger.Destroy(debugger)
    # Terminate the debug session
    lldb.SBDebugger.Terminate()
```

### Example

Now your python scripts are ready to import the lldb module. Below is a python
script that will launch a program from the current working directory called
`a.out`, set a breakpoint at `main`, and then run and hit the breakpoint, and
print the process, thread and frame objects if the process stopped:

```python3
#!/usr/bin/env python3

import lldb
import os

def disassemble_instructions(insts):
    for i in insts:
        print(i)

# Set the path to the executable to debug
exe = "./a.out"

# Create a new debugger instance
debugger = lldb.SBDebugger.Create()

# When we step or continue, don't return from the function until the process
# stops. Otherwise we would have to handle the process events ourselves which, while doable is
# a little tricky.  We do this by setting the async mode to false.
debugger.SetAsync(False)

# Create a target from a file and arch
print("Creating a target for '%s'" % exe)

target = debugger.CreateTargetWithFileAndArch(exe, lldb.LLDB_ARCH_DEFAULT)

if target:
    # If the target is valid set a breakpoint at main
    main_bp = target.BreakpointCreateByName(
        "main", target.GetExecutable().GetFilename()
    )

    print(main_bp)

    # Launch the process. Since we specified synchronous mode, we won't return
    # from this function until we hit the breakpoint at main
    process = target.LaunchSimple(None, None, os.getcwd())

    # Make sure the launch went ok
    if process:
        # Print some simple process info
        state = process.GetState()
        print(process)
        if state == lldb.eStateStopped:
            # Get the first thread
            thread = process.GetThreadAtIndex(0)
            if thread:
                # Print some simple thread info
                print(thread)
                # Get the first frame
                frame = thread.GetFrameAtIndex(0)
                if frame:
                    # Print some simple frame info
                    print(frame)
                    function = frame.GetFunction()
                    # See if we have debug info (a function)
                    if function:
                        # We do have a function, print some info for the function
                        print(function)
                        # Now get all instructions for this function and print them
                        insts = function.GetInstructions(target)
                        disassemble_instructions(insts)
                    else:
                        # See if we have a symbol in the symbol table for where we stopped
                        symbol = frame.GetSymbol()
                        if symbol:
                            # We do have a symbol, print some info for the symbol
                            print(symbol)
```