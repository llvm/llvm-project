%feature("docstring",
"Represents a logical breakpoint and its associated settings.

For example (from test/functionalities/breakpoint/breakpoint_ignore_count/
TestBreakpointIgnoreCount.py),::

    def breakpoint_ignore_count_python(self):
        '''Use Python APIs to set breakpoint ignore count.'''
        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target by the debugger.
        target = self.dbg.CreateTarget(exe)
        self.assertTrue(target, VALID_TARGET)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')
        self.assertTrue(breakpoint and
                        breakpoint.GetNumLocations() == 1,
                        VALID_BREAKPOINT)

        # Get the breakpoint location from breakpoint after we verified that,
        # indeed, it has one location.
        location = breakpoint.GetLocationAtIndex(0)
        self.assertTrue(location and
                        location.IsEnabled(),
                        VALID_BREAKPOINT_LOCATION)

        # Set the ignore count on the breakpoint location.
        location.SetIgnoreCount(2)
        self.assertTrue(location.GetIgnoreCount() == 2,
                        'SetIgnoreCount() works correctly')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())
        self.assertTrue(process, PROCESS_IS_VALID)

        # Frame#0 should be on main.c:37, frame#1 should be on main.c:25, and
        # frame#2 should be on main.c:48.
        #lldbutil.print_stacktraces(process)
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        self.assertTrue(thread != None, 'There should be a thread stopped due to breakpoint')
        frame0 = thread.GetFrameAtIndex(0)
        frame1 = thread.GetFrameAtIndex(1)
        frame2 = thread.GetFrameAtIndex(2)
        self.assertTrue(frame0.GetLineEntry().GetLine() == self.line1 and
                        frame1.GetLineEntry().GetLine() == self.line3 and
                        frame2.GetLineEntry().GetLine() == self.line4,
                        STOPPED_DUE_TO_BREAKPOINT_IGNORE_COUNT)

        # The hit count for the breakpoint should be 3.
        self.assertTrue(breakpoint.GetHitCount() == 3)

        process.Continue()

SBBreakpoint supports breakpoint location iteration, for example,::

    for bl in breakpoint:
        print('breakpoint location load addr: %s' % hex(bl.GetLoadAddress()))
        print('breakpoint location condition: %s' % hex(bl.GetCondition()))

and rich comparison methods which allow the API program to use,::

    if aBreakpoint == bBreakpoint:
        ...

to compare two breakpoints for equality."
) lldb::SBBreakpoint;

%feature("docstring", "
    The breakpoint stops only if the condition expression evaluates to true."
) lldb::SBBreakpoint::SetCondition;

%feature("docstring", "
    Get the condition expression for the breakpoint."
) lldb::SBBreakpoint::GetCondition;

%feature("docstring", "
    Set the name of the script function to be called when the breakpoint is hit."
) lldb::SBBreakpoint::SetScriptCallbackFunction;

%feature("docstring", "
    Set the name of the script function to be called when the breakpoint is hit.
    To use this variant, the function should take (frame, bp_loc, extra_args, internal_dict) and
    when the breakpoint is hit the extra_args will be passed to the callback function."
) lldb::SBBreakpoint::SetScriptCallbackFunction;

%feature("docstring", "
    Provide the body for the script function to be called when the breakpoint is hit.
    The body will be wrapped in a function, which be passed two arguments:
    'frame' - which holds the bottom-most SBFrame of the thread that hit the breakpoint
    'bpno'  - which is the SBBreakpointLocation to which the callback was attached.

    The error parameter is currently ignored, but will at some point hold the Python
    compilation diagnostics.
    Returns true if the body compiles successfully, false if not."
) lldb::SBBreakpoint::SetScriptCallbackBody;


%feature("docstring",
"Represents a list of :py:class:`SBBreakpoint`."
) lldb::SBBreakpointList;
