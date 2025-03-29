%feature("docstring",
"Represents an instance of watchpoint for a specific target program.

A watchpoint is determined by the address and the byte size that resulted in
this particular instantiation.  Each watchpoint has its settable options.

See also :py:class:`SBTarget.watchpoint_iter()` for example usage of iterating through the
watchpoints of the target."
) lldb::SBWatchpoint;

%feature("docstring", "
    Deprecated.  Previously: Return the hardware index of the 
    watchpoint register.  Now: -1 is always returned."
) lldb::SBWatchpoint::GetHardwareIndex;

%feature("docstring", "
    Get the condition expression for the watchpoint."
) lldb::SBWatchpoint::GetCondition;

%feature("docstring", "
    The watchpoint stops only if the condition expression evaluates to true."
) lldb::SBWatchpoint::SetCondition;

%feature("docstring", "
    Returns the type recorded when the watchpoint was created. For variable
    watchpoints it is the type of the watched variable. For expression
    watchpoints it is the type of the provided expression."
) lldb::SBWatchpoint::GetType;

%feature("docstring", "
    Returns the kind of value that was watched when the watchpoint was created.
    Returns one of the following eWatchPointValueKindVariable,
    eWatchPointValueKindExpression, eWatchPointValueKindInvalid.
    "
) lldb::SBWatchpoint::GetWatchValueKind;

%feature("docstring", "
    Get the spec for the watchpoint. For variable watchpoints this is the name
    of the variable. For expression watchpoints it is empty
    (may change in the future)."
) lldb::SBWatchpoint::GetWatchSpec;

%feature("docstring", "
    Returns true if the watchpoint is watching reads. Returns false otherwise."
) lldb::SBWatchpoint::IsWatchingReads;

%feature("docstring", "
    Returns true if the watchpoint is watching writes. Returns false otherwise."
) lldb::SBWatchpoint::IsWatchingWrites;
