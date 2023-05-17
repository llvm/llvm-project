%feature("docstring",
"A context object that provides access to core debugger entities.

Many debugger functions require a context when doing lookups. This class
provides a common structure that can be used as the result of a query that
can contain a single result.

For example, ::

        exe = os.path.join(os.getcwd(), 'a.out')

        # Create a target for the debugger.
        target = self.dbg.CreateTarget(exe)

        # Now create a breakpoint on main.c by name 'c'.
        breakpoint = target.BreakpointCreateByName('c', 'a.out')

        # Now launch the process, and do not stop at entry point.
        process = target.LaunchSimple(None, None, os.getcwd())

        # The inferior should stop on 'c'.
        from lldbutil import get_stopped_thread
        thread = get_stopped_thread(process, lldb.eStopReasonBreakpoint)
        frame0 = thread.GetFrameAtIndex(0)

        # Now get the SBSymbolContext from this frame.  We want everything. :-)
        context = frame0.GetSymbolContext(lldb.eSymbolContextEverything)

        # Get the module.
        module = context.GetModule()
        ...

        # And the compile unit associated with the frame.
        compileUnit = context.GetCompileUnit()
        ...
"
) lldb::SBSymbolContext;
