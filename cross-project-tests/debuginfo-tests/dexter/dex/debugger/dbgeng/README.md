# Debugger Engine backend

This directory contains the Dexter backend for the Windows Debugger Engine
(DbgEng), which powers tools such as WinDbg and CDB.

## Overview

DbgEng is available as a collection of unregistered COM-"like" objects that
one accesses by calling DebugCreate in DbgEng.dll. The unregistered nature
means normal COM tooling can't access them; as a result, this backend uses
ctypes to describe the COM objects and call their methods.

This is obviously not a huge amount of fun; on the other hand, COM has
maintained ABI compatible interfaces for decades, and nothing is for free.

The dexter backend follows the same formula as others; it creates a process
and breaks on "main", then steps through the program, observing states and
stack frames along the way.

## Implementation details

This backend uses a mixture of both APIs for accessing information, and the
direct command-string interface to DbgEng for performing some actions. We
have to use the DbgEng stepping interface, or we would effectively be
building a new debugger, but certain things (like enabling source-line
stepping) only seem to be possible from the command interface.

Each segment of debugger responsibility has its own COM object: Client,
Control, Symbols, SymbolGroups, Breakpoint, SystemObjects. In this python
wrapper, each COM object gets a python object wrapping it. COM methods
that are relevant to our interests have a python method that wraps the COM
one and performs data marshalling. Some additional helper methods are added
to the python objects to extract data.

The majority of the work occurs in setup.py and probe_process.py. The
former contains routines to launch a process and attach the debugger to
it, while the latter extracts as much information as possible from a
stopped process, returning a list of stack frames with associated variable
information.

## Sharp edges

On process startup, we set a breakpoint on main and then continue running
to it. This has the potential to never complete -- although of course,
there's no guarantee that the debuggee will ever do anything anyway.

There doesn't appear to be a way to instruct DbgEng to "step into" a
function call, thus after reaching main, we scan the module for all
functions with line numbers in the source directory, and put breakpoints
on them. An alternative implementation would be putting breakpoints on
every known line number.

Finally, it's unclear whether arbitrary expressions can be evaluated in
arbitrary stack frames, although this isn't something that Dexter currently
supports.
 
