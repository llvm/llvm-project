# Accessing Script Documentation

The LLDB API is contained in a python module named lldb. A useful resource when
writing Python extensions is the lldb Python classes reference guide.

The documentation is also accessible in an interactive debugger session with
the following command:

```python3
(lldb) script help(lldb)
   Help on package lldb:

   NAME
      lldb - The lldb module contains the public APIs for Python binding.

   FILE
      /System/Library/PrivateFrameworks/LLDB.framework/Versions/A/Resources/Python/lldb/__init__.py

   DESCRIPTION
...
```

You can also get help using a module class name. The full API that is exposed
for that class will be displayed in a man page style window. Below we want to
get help on the lldb.SBFrame class:

```python3
(lldb) script help(lldb.SBFrame)
   Help on class SBFrame in module lldb:

   class SBFrame(builtins.object)
    |  SBFrame(*args)
    |  
    |  Represents one of the stack frames associated with a thread.
    |  
    |  SBThread contains SBFrame(s). For example (from test/lldbutil.py), ::
    |  
    |      def print_stacktrace(thread, string_buffer = False):
    |          '''Prints a simple stack trace of this thread.'''
...
```

Or you can get help using any python object, here we use the lldb.process
object which is a global variable in the lldb module which represents the
currently selected process:

```python3
(lldb) script help(lldb.process)
   Help on SBProcess in module lldb object:

   class SBProcess(builtins.object)
    |  SBProcess(*args)
    |  
    |  Represents the process associated with the target program.
    |  
    |  SBProcess supports thread iteration. For example (from test/lldbutil.py), ::
    |  
    |      # ==================================================
    |      # Utility functions related to Threads and Processes
    |      # ==================================================
...
```