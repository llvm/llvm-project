%feature("docstring",
"Represents an instance of watchpoint for a specific target program.

A watchpoint is determined by the address and the byte size that resulted in
this particular instantiation.  Each watchpoint has its settable options.

See also :py:class:`SBTarget.watchpoint_iter()` for example usage of iterating through the
watchpoints of the target."
) lldb::SBWatchpoint;

%feature("docstring", "
    With -1 representing an invalid hardware index."
) lldb::SBWatchpoint::GetHardwareIndex;

%feature("docstring", "
    Get the condition expression for the watchpoint."
) lldb::SBWatchpoint::GetCondition;

%feature("docstring", "
    The watchpoint stops only if the condition expression evaluates to true."
) lldb::SBWatchpoint::SetCondition;
