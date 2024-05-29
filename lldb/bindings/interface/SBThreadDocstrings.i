%feature("docstring",
"Represents a thread of execution. :py:class:`SBProcess` contains SBThread(s).

SBThreads can be referred to by their ID, which maps to the system specific thread
identifier, or by IndexID.  The ID may or may not be unique depending on whether the
system reuses its thread identifiers.  The IndexID is a monotonically increasing identifier
that will always uniquely reference a particular thread, and when that thread goes
away it will not be reused.

SBThread supports frame iteration. For example (from test/python_api/
lldbutil/iter/TestLLDBIterator.py), ::

        from lldbutil import print_stacktrace
        stopped_due_to_breakpoint = False
        for thread in process:
            if self.TraceOn():
                print_stacktrace(thread)
            ID = thread.GetThreadID()
            if thread.GetStopReason() == lldb.eStopReasonBreakpoint:
                stopped_due_to_breakpoint = True
            for frame in thread:
                self.assertTrue(frame.GetThread().GetThreadID() == ID)
                if self.TraceOn():
                    print frame

        self.assertTrue(stopped_due_to_breakpoint)

See also :py:class:`SBFrame` ."
) lldb::SBThread;

%feature("docstring", "
    Get the number of words associated with the stop reason.
    See also GetStopReasonDataAtIndex()."
) lldb::SBThread::GetStopReasonDataCount;

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
) lldb::SBThread::GetStopReasonDataAtIndex;

%feature("docstring", "
    Collects a thread's stop reason extended information dictionary and prints it
    into the SBStream in a JSON format. The format of this JSON dictionary depends
    on the stop reason and is currently used only for instrumentation plugins."
) lldb::SBThread::GetStopReasonExtendedInfoAsJSON;

%feature("docstring", "
    Returns a collection of historical stack traces that are significant to the
    current stop reason. Used by ThreadSanitizer, where we provide various stack
    traces that were involved in a data race or other type of detected issue."
) lldb::SBThread::GetStopReasonExtendedBacktraces;

%feature("docstring", "
    Pass only an (int)length and expect to get a Python string describing the
    stop reason."
) lldb::SBThread::GetStopDescription;

%feature("docstring", "
    Returns a unique thread identifier (type lldb::tid_t, typically a 64-bit type)
    for the current SBThread that will remain constant throughout the thread's
    lifetime in this process and will not be reused by another thread during this
    process lifetime.  On Mac OS X systems, this is a system-wide unique thread
    identifier; this identifier is also used by other tools like sample which helps
    to associate data from those tools with lldb.  See related GetIndexID."
) lldb::SBThread::GetThreadID;

%feature("docstring", "
    Return the index number for this SBThread.  The index number is the same thing
    that a user gives as an argument to 'thread select' in the command line lldb.
    These numbers start at 1 (for the first thread lldb sees in a debug session)
    and increments up throughout the process lifetime.  An index number will not be
    reused for a different thread later in a process - thread 1 will always be
    associated with the same thread.  See related GetThreadID.
    This method returns a uint32_t index number, takes no arguments."
) lldb::SBThread::GetIndexID;

%feature("docstring", "
    Return the queue name associated with this thread, if any, as a str.
    For example, with a libdispatch (aka Grand Central Dispatch) queue."
) lldb::SBThread::GetQueueName;

%feature("docstring", "
    Return the dispatch_queue_id for this thread, if any, as a lldb::queue_id_t.
    For example, with a libdispatch (aka Grand Central Dispatch) queue."
) lldb::SBThread::GetQueueID;

%feature("docstring", "
    Takes a path string and a SBStream reference as parameters, returns a bool.
    Collects the thread's 'info' dictionary from the remote system, uses the path
    argument to descend into the dictionary to an item of interest, and prints
    it into the SBStream in a natural format.  Return bool is to indicate if
    anything was printed into the stream (true) or not (false)."
) lldb::SBThread::GetInfoItemByPathAsString;

%feature("docstring", "
    Return the SBQueue for this thread.  If this thread is not currently associated
    with a libdispatch queue, the SBQueue object's IsValid() method will return false.
    If this SBThread is actually a HistoryThread, we may be able to provide QueueID
    and QueueName, but not provide an SBQueue.  Those individual attributes may have
    been saved for the HistoryThread without enough information to reconstitute the
    entire SBQueue at that time.
    This method takes no arguments, returns an SBQueue."
) lldb::SBThread::GetQueue;

%feature("docstring",
    "Do a source level single step over in the currently selected thread."
) lldb::SBThread::StepOver;

%feature("docstring", "
    Step the current thread from the current source line to the line given by end_line, stopping if
    the thread steps into the function given by target_name.  If target_name is None, then stepping will stop
    in any of the places we would normally stop."
) lldb::SBThread::StepInto;

%feature("docstring",
    "Step out of the currently selected thread."
) lldb::SBThread::StepOut;

%feature("docstring",
    "Step out of the specified frame."
) lldb::SBThread::StepOutOfFrame;

%feature("docstring",
    "Do an instruction level single step in the currently selected thread."
) lldb::SBThread::StepInstruction;

%feature("docstring", "
    Force a return from the frame passed in (and any frames younger than it)
    without executing any more code in those frames.  If return_value contains
    a valid SBValue, that will be set as the return value from frame.  Note, at
    present only scalar return values are supported."
) lldb::SBThread::ReturnFromFrame;

%feature("docstring", "
    Unwind the stack frames from the innermost expression evaluation.
    This API is equivalent to 'thread return -x'."
) lldb::SBThread::UnwindInnermostExpression;

%feature("docstring", "
    LLDB currently supports process centric debugging which means when any
    thread in a process stops, all other threads are stopped. The Suspend()
    call here tells our process to suspend a thread and not let it run when
    the other threads in a process are allowed to run. So when
    SBProcess::Continue() is called, any threads that aren't suspended will
    be allowed to run. If any of the SBThread functions for stepping are
    called (StepOver, StepInto, StepOut, StepInstruction, RunToAddres), the
    thread will now be allowed to run and these functions will simply return.

    Eventually we plan to add support for thread centric debugging where
    each thread is controlled individually and each thread would broadcast
    its state, but we haven't implemented this yet.

    Likewise the SBThread::Resume() call will again allow the thread to run
    when the process is continued.

    Suspend() and Resume() functions are not currently reference counted, if
    anyone has the need for them to be reference counted, please let us
    know."
) lldb::SBThread::Suspend;

%feature("docstring", "
    Get the description strings for this thread that match what the
    lldb driver will present, using the thread-format (stop_format==false)
    or thread-stop-format (stop_format = true)."
) lldb::SBThread::GetDescription;

%feature("docstring","
    Given an argument of str to specify the type of thread-origin extended
    backtrace to retrieve, query whether the origin of this thread is
    available.  An SBThread is retured; SBThread.IsValid will return true
    if an extended backtrace was available.  The returned SBThread is not
    a part of the SBProcess' thread list and it cannot be manipulated like
    normal threads -- you cannot step or resume it, for instance -- it is
    intended to used primarily for generating a backtrace.  You may request
    the returned thread's own thread origin in turn."
) lldb::SBThread::GetExtendedBacktraceThread;

%feature("docstring","
    If this SBThread is an ExtendedBacktrace thread, get the IndexID of the
    original thread that this ExtendedBacktrace thread represents, if
    available.  The thread that was running this backtrace in the past may
    not have been registered with lldb's thread index (if it was created,
    did its work, and was destroyed without lldb ever stopping execution).
    In that case, this ExtendedBacktrace thread's IndexID will be returned."
) lldb::SBThread::GetExtendedBacktraceOriginatingIndexID;

%feature("docstring","
    Returns an SBValue object represeting the current exception for the thread,
    if there is any. Currently, this works for Obj-C code and returns an SBValue
    representing the NSException object at the throw site or that's currently
    being processes."
) lldb::SBThread::GetCurrentException;

%feature("docstring","
    Returns a historical (fake) SBThread representing the stack trace of an
    exception, if there is one for the thread. Currently, this works for Obj-C
    code, and can retrieve the throw-site backtrace of an NSException object
    even when the program is no longer at the throw site."
) lldb::SBThread::GetCurrentExceptionBacktrace;

%feature("docstring","
    lldb may be able to detect that function calls should not be executed
    on a given thread at a particular point in time.  It is recommended that
    this is checked before performing an inferior function call on a given
    thread."
) lldb::SBThread::SafeToCallFunctions;

%feature("docstring","
    Returns a SBValue object representing the siginfo for the current signal.
    "
) lldb::SBThread::GetSiginfo;
