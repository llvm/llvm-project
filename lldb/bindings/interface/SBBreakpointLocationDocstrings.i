%feature("docstring",
"Represents one unique instance (by address) of a logical breakpoint.

A breakpoint location is defined by the breakpoint that produces it,
and the address that resulted in this particular instantiation.
Each breakpoint location has its settable options.

:py:class:`SBBreakpoint` contains SBBreakpointLocation(s). See docstring of SBBreakpoint
for retrieval of an SBBreakpointLocation from an SBBreakpoint."
) lldb::SBBreakpointLocation;

%feature("docstring", "
    The breakpoint location stops only if the condition expression evaluates
    to true.") lldb::SBBreakpointLocation::SetCondition;

%feature("docstring", "
    Get the condition expression for the breakpoint location."
) lldb::SBBreakpointLocation::GetCondition;

%feature("docstring", "
    Set the callback to the given Python function name.
    The function takes three arguments (frame, bp_loc, internal_dict)."
) lldb::SBBreakpointLocation::SetScriptCallbackFunction;

%feature("docstring", "
    Set the name of the script function to be called when the breakpoint is hit.
    To use this variant, the function should take (frame, bp_loc, extra_args, internal_dict) and
    when the breakpoint is hit the extra_args will be passed to the callback function."
) lldb::SBBreakpointLocation::SetScriptCallbackFunction;

%feature("docstring", "
    Provide the body for the script function to be called when the breakpoint location is hit.
    The body will be wrapped in a function, which be passed two arguments:
    'frame' - which holds the bottom-most SBFrame of the thread that hit the breakpoint
    'bpno'  - which is the SBBreakpointLocation to which the callback was attached.

    The error parameter is currently ignored, but will at some point hold the Python
    compilation diagnostics.
    Returns true if the body compiles successfully, false if not."
) lldb::SBBreakpointLocation::SetScriptCallbackBody;
