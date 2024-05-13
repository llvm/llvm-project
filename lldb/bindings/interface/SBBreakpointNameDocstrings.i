%feature("docstring",
"Represents a breakpoint name registered in a given :py:class:`SBTarget`.

Breakpoint names provide a way to act on groups of breakpoints.  When you add a
name to a group of breakpoints, you can then use the name in all the command
line lldb commands for that name.  You can also configure the SBBreakpointName
options and those options will be propagated to any :py:class:`SBBreakpoint` s currently
using that name.  Adding a name to a breakpoint will also apply any of the
set options to that breakpoint.

You can also set permissions on a breakpoint name to disable listing, deleting
and disabling breakpoints.  That will disallow the given operation for breakpoints
except when the breakpoint is mentioned by ID.  So for instance deleting all the
breakpoints won't delete breakpoints so marked."
) lldb::SBBreakpointName;
