%feature("docstring",
"Represents a plan for the execution control of a given thread.

See also :py:class:`SBThread` and :py:class:`SBFrame`."
) lldb::SBThreadPlan;

%feature("docstring", "
    Get the number of words associated with the stop reason.
    See also GetStopReasonDataAtIndex()."
) lldb::SBThreadPlan::GetStopReasonDataCount;

%feature("docstring", "
    Get information associated with a stop reason.

    Breakpoint stop reasons will have data that consists of pairs of
    breakpoint IDs followed by the breakpoint location IDs (they always come
    in pairs).

    Stop Reason              Count Data Type
    ======================== ===== =========================================
    eStopReasonNone          0
    eStopReasonTrace         0
    eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    eStopReasonWatchpoint    1     watchpoint id
    eStopReasonSignal        1     unix signal number
    eStopReasonException     N     exception data
    eStopReasonExec          0
    eStopReasonFork          1     pid of the child process
    eStopReasonVFork         1     pid of the child process
    eStopReasonVForkDone     0
    eStopReasonPlanComplete  0"
) lldb::SBThreadPlan::GetStopReasonDataAtIndex;

%feature("docstring", "Return whether this plan will ask to stop other threads when it runs."
) lldb::SBThreadPlan::GetStopOthers;

%feature("docstring", "Set whether this plan will ask to stop other threads when it runs."
) lldb::SBThreadPlan::SetStopOthers;
