# Extending Target Stop-Hooks

Stop hooks fire whenever the process stops just before control is returned to the
user.  Stop hooks can either be a set of lldb command-line commands, or can
be implemented by a suitably defined Python class.  The Python-based stop-hooks
can also be passed as a set of -key -value pairs when they are added, and those
will get packaged up into a `SBStructuredData` Dictionary and passed to the
constructor of the Python object managing the stop hook.  This allows for
parameterization of the stop hooks.

To add a Python-based stop hook, first define a class with the following methods:

| Name | Arguments | Description |
|------|-----------|-------------|
| `__init__` | `target: lldb.SBTarget` `extra_args: lldb.SBStructuredData` | This is the constructor for the new stop-hook. `target` is the SBTarget to which the stop hook is added. `extra_args` is an SBStructuredData object that the user can pass in when creating instances of this breakpoint. It is not required, but allows for reuse of stop-hook classes. |
| `handle_stop` | `exe_ctx: lldb.SBExecutionContext` `stream: lldb.SBStream` | This is the called when the target stops. `exe_ctx` argument will be filled with the current stop point for which the stop hook is being evaluated. `stream` an lldb.SBStream, anything written to this stream will be written to the debugger console. The return value is a "Should Stop" vote from this thread. If the method returns either True or no return this thread votes to stop. If it returns False, then the thread votes to continue after all the stop-hooks are evaluated. Note, the --auto-continue flag to 'target stop-hook add' overrides a True return value from the method. |

To use this class in lldb, run the command:

```
(lldb) command script import MyModule.py
(lldb) target stop-hook add -P MyModule.MyStopHook -k first -v 1 -k second -v 2
```

where `MyModule.py` is the file containing the class definition `MyStopHook`.