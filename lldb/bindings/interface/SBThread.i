//===-- SWIG Interface for SBThread -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

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
) SBThread;
class SBThread
{
public:
    //------------------------------------------------------------------
    // Broadcaster bits.
    //------------------------------------------------------------------
    enum
    {
        eBroadcastBitStackChanged           = (1 << 0),
        eBroadcastBitThreadSuspended        = (1 << 1),
        eBroadcastBitThreadResumed          = (1 << 2),
        eBroadcastBitSelectedFrameChanged   = (1 << 3),
        eBroadcastBitThreadSelected         = (1 << 4)
    };


    SBThread ();

    SBThread (const lldb::SBThread &thread);

   ~SBThread();

    static const char *
    GetBroadcasterClassName ();

    static bool
    EventIsThreadEvent (const SBEvent &event);

    static SBFrame
    GetStackFrameFromEvent (const SBEvent &event);

    static SBThread
    GetThreadFromEvent (const SBEvent &event);

    bool
    IsValid() const;

    explicit operator bool() const;

    void
    Clear ();

    lldb::StopReason
    GetStopReason();

    %feature("docstring", "
    Get the number of words associated with the stop reason.
    See also GetStopReasonDataAtIndex().") GetStopReasonDataCount;
    size_t
    GetStopReasonDataCount();

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
    eStopReasonPlanComplete  0") GetStopReasonDataAtIndex;
    uint64_t
    GetStopReasonDataAtIndex(uint32_t idx);

    %feature("autodoc", "
    Collects a thread's stop reason extended information dictionary and prints it
    into the SBStream in a JSON format. The format of this JSON dictionary depends
    on the stop reason and is currently used only for instrumentation plugins.") GetStopReasonExtendedInfoAsJSON;
    bool
    GetStopReasonExtendedInfoAsJSON (lldb::SBStream &stream);

    %feature("autodoc", "
    Returns a collection of historical stack traces that are significant to the
    current stop reason. Used by ThreadSanitizer, where we provide various stack
    traces that were involved in a data race or other type of detected issue.") GetStopReasonExtendedBacktraces;
    SBThreadCollection
    GetStopReasonExtendedBacktraces (InstrumentationRuntimeType type);


    %feature("autodoc", "
    Pass only an (int)length and expect to get a Python string describing the
    stop reason.") GetStopDescription;
    size_t
    GetStopDescription (char *dst_or_null, size_t dst_len);

    SBValue
    GetStopReturnValue ();

    %feature("autodoc", "
    Returns a unique thread identifier (type lldb::tid_t, typically a 64-bit type)
    for the current SBThread that will remain constant throughout the thread's
    lifetime in this process and will not be reused by another thread during this
    process lifetime.  On Mac OS X systems, this is a system-wide unique thread
    identifier; this identifier is also used by other tools like sample which helps
    to associate data from those tools with lldb.  See related GetIndexID.")
    GetThreadID;
    lldb::tid_t
    GetThreadID () const;

    %feature("autodoc", "
    Return the index number for this SBThread.  The index number is the same thing
    that a user gives as an argument to 'thread select' in the command line lldb.
    These numbers start at 1 (for the first thread lldb sees in a debug session)
    and increments up throughout the process lifetime.  An index number will not be
    reused for a different thread later in a process - thread 1 will always be
    associated with the same thread.  See related GetThreadID.
    This method returns a uint32_t index number, takes no arguments.")
    GetIndexID;
    uint32_t
    GetIndexID () const;

    const char *
    GetName () const;

    %feature("autodoc", "
    Return the queue name associated with this thread, if any, as a str.
    For example, with a libdispatch (aka Grand Central Dispatch) queue.") GetQueueName;

    const char *
    GetQueueName() const;

    %feature("autodoc", "
    Return the dispatch_queue_id for this thread, if any, as a lldb::queue_id_t.
    For example, with a libdispatch (aka Grand Central Dispatch) queue.") GetQueueID;

    lldb::queue_id_t
    GetQueueID() const;

    %feature("docstring", "
    Takes a path string and a SBStream reference as parameters, returns a bool.
    Collects the thread's 'info' dictionary from the remote system, uses the path
    argument to descend into the dictionary to an item of interest, and prints
    it into the SBStream in a natural format.  Return bool is to indicate if
    anything was printed into the stream (true) or not (false).") GetInfoItemByPathAsString;

    bool
    GetInfoItemByPathAsString (const char *path, lldb::SBStream &strm);

    %feature("autodoc", "
    Return the SBQueue for this thread.  If this thread is not currently associated
    with a libdispatch queue, the SBQueue object's IsValid() method will return false.
    If this SBThread is actually a HistoryThread, we may be able to provide QueueID
    and QueueName, but not provide an SBQueue.  Those individual attributes may have
    been saved for the HistoryThread without enough information to reconstitute the
    entire SBQueue at that time.
    This method takes no arguments, returns an SBQueue.") GetQueue;

    lldb::SBQueue
    GetQueue () const;

    void
    StepOver (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    %feature("autodoc",
    "Do a source level single step over in the currently selected thread.") StepOver;
    void
    StepOver (lldb::RunMode stop_other_threads, SBError &error);

    void
    StepInto (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepInto (const char *target_name, lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    %feature("autodoc", "
    Step the current thread from the current source line to the line given by end_line, stopping if
    the thread steps into the function given by target_name.  If target_name is None, then stepping will stop
    in any of the places we would normally stop.") StepInto;
    void
    StepInto (const char *target_name,
              uint32_t end_line,
              SBError &error,
              lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepOut ();

    %feature("autodoc",
    "Step out of the currently selected thread.") StepOut;
    void
    StepOut (SBError &error);

    void
    StepOutOfFrame (SBFrame &frame);

    %feature("autodoc",
    "Step out of the specified frame.") StepOutOfFrame;
    void
    StepOutOfFrame (SBFrame &frame, SBError &error);

    void
    StepInstruction(bool step_over);

    %feature("autodoc",
    "Do an instruction level single step in the currently selected thread.") StepInstruction;
    void
    StepInstruction(bool step_over, SBError &error);

    SBError
    StepOverUntil (lldb::SBFrame &frame,
                   lldb::SBFileSpec &file_spec,
                   uint32_t line);

    SBError
    StepUsingScriptedThreadPlan (const char *script_class_name);

    SBError
    StepUsingScriptedThreadPlan (const char *script_class_name, bool resume_immediately);

    SBError
    StepUsingScriptedThreadPlan(const char *script_class_name,
                                lldb::SBStructuredData &args_data,
                                bool resume_immediately);

    SBError
    JumpToLine (lldb::SBFileSpec &file_spec, uint32_t line);

    void
    RunToAddress (lldb::addr_t addr);

    void
    RunToAddress (lldb::addr_t addr, SBError &error);

    %feature("autodoc", "
    Force a return from the frame passed in (and any frames younger than it)
    without executing any more code in those frames.  If return_value contains
    a valid SBValue, that will be set as the return value from frame.  Note, at
    present only scalar return values are supported.") ReturnFromFrame;

    SBError
    ReturnFromFrame (SBFrame &frame, SBValue &return_value);

    %feature("autodoc", "
    Unwind the stack frames from the innermost expression evaluation.
    This API is equivalent to 'thread return -x'.") UnwindInnermostExpression;

    SBError
    UnwindInnermostExpression();

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
    know.") Suspend;
    bool
    Suspend();

    bool
    Suspend(SBError &error);

    bool
    Resume ();

    bool
    Resume (SBError &error);

    bool
    IsSuspended();

    bool
    IsStopped();

    uint32_t
    GetNumFrames ();

    lldb::SBFrame
    GetFrameAtIndex (uint32_t idx);

    lldb::SBFrame
    GetSelectedFrame ();

    lldb::SBFrame
    SetSelectedFrame (uint32_t frame_idx);

    lldb::SBProcess
    GetProcess ();

    bool
    GetDescription (lldb::SBStream &description) const;

    %feature("docstring", "
    Get the description strings for this thread that match what the
    lldb driver will present, using the thread-format (stop_format==false)
    or thread-stop-format (stop_format = true).") GetDescription;
    bool GetDescription(lldb::SBStream &description, bool stop_format) const;

    bool
    GetStatus (lldb::SBStream &status) const;

    bool
    operator == (const lldb::SBThread &rhs) const;

    bool
    operator != (const lldb::SBThread &rhs) const;

    %feature("autodoc","
    Given an argument of str to specify the type of thread-origin extended
    backtrace to retrieve, query whether the origin of this thread is
    available.  An SBThread is retured; SBThread.IsValid will return true
    if an extended backtrace was available.  The returned SBThread is not
    a part of the SBProcess' thread list and it cannot be manipulated like
    normal threads -- you cannot step or resume it, for instance -- it is
    intended to used primarily for generating a backtrace.  You may request
    the returned thread's own thread origin in turn.") GetExtendedBacktraceThread;
    lldb::SBThread
    GetExtendedBacktraceThread (const char *type);

    %feature("autodoc","
    Takes no arguments, returns a uint32_t.
    If this SBThread is an ExtendedBacktrace thread, get the IndexID of the
    original thread that this ExtendedBacktrace thread represents, if
    available.  The thread that was running this backtrace in the past may
    not have been registered with lldb's thread index (if it was created,
    did its work, and was destroyed without lldb ever stopping execution).
    In that case, this ExtendedBacktrace thread's IndexID will be returned.") GetExtendedBacktraceOriginatingIndexID;
    uint32_t
    GetExtendedBacktraceOriginatingIndexID();

    %feature("autodoc","
    Returns an SBValue object represeting the current exception for the thread,
    if there is any. Currently, this works for Obj-C code and returns an SBValue
    representing the NSException object at the throw site or that's currently
    being processes.") GetCurrentException;
    lldb::SBValue
    GetCurrentException();

    %feature("autodoc","
    Returns a historical (fake) SBThread representing the stack trace of an
    exception, if there is one for the thread. Currently, this works for Obj-C
    code, and can retrieve the throw-site backtrace of an NSException object
    even when the program is no longer at the throw site.") GetCurrentExceptionBacktrace;
    lldb::SBThread
    GetCurrentExceptionBacktrace();

    %feature("autodoc","
    Takes no arguments, returns a bool.
    lldb may be able to detect that function calls should not be executed
    on a given thread at a particular point in time.  It is recommended that
    this is checked before performing an inferior function call on a given
    thread.") SafeToCallFunctions;
    bool
    SafeToCallFunctions ();

    STRING_EXTENSION(SBThread)

#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all frames in a lldb.SBThread object.'''
            return lldb_iter(self, 'GetNumFrames', 'GetFrameAtIndex')

        def __len__(self):
            '''Return the number of frames in a lldb.SBThread object.'''
            return self.GetNumFrames()

        class frames_access(object):
            '''A helper object that will lazily hand out frames for a thread when supplied an index.'''
            def __init__(self, sbthread):
                self.sbthread = sbthread

            def __len__(self):
                if self.sbthread:
                    return int(self.sbthread.GetNumFrames())
                return 0

            def __getitem__(self, key):
                if type(key) is int and key < self.sbthread.GetNumFrames():
                    return self.sbthread.GetFrameAtIndex(key)
                return None

        def get_frames_access_object(self):
            '''An accessor function that returns a frames_access() object which allows lazy frame access from a lldb.SBThread object.'''
            return self.frames_access (self)

        def get_thread_frames(self):
            '''An accessor function that returns a list() that contains all frames in a lldb.SBThread object.'''
            frames = []
            for frame in self:
                frames.append(frame)
            return frames

        id = property(GetThreadID, None, doc='''A read only property that returns the thread ID as an integer.''')
        idx = property(GetIndexID, None, doc='''A read only property that returns the thread index ID as an integer. Thread index ID values start at 1 and increment as threads come and go and can be used to uniquely identify threads.''')
        return_value = property(GetStopReturnValue, None, doc='''A read only property that returns an lldb object that represents the return value from the last stop (lldb.SBValue) if we just stopped due to stepping out of a function.''')
        process = property(GetProcess, None, doc='''A read only property that returns an lldb object that represents the process (lldb.SBProcess) that owns this thread.''')
        num_frames = property(GetNumFrames, None, doc='''A read only property that returns the number of stack frames in this thread as an integer.''')
        frames = property(get_thread_frames, None, doc='''A read only property that returns a list() of lldb.SBFrame objects for all frames in this thread.''')
        frame = property(get_frames_access_object, None, doc='''A read only property that returns an object that can be used to access frames as an array ("frame_12 = lldb.thread.frame[12]").''')
        name = property(GetName, None, doc='''A read only property that returns the name of this thread as a string.''')
        queue = property(GetQueueName, None, doc='''A read only property that returns the dispatch queue name of this thread as a string.''')
        queue_id = property(GetQueueID, None, doc='''A read only property that returns the dispatch queue id of this thread as an integer.''')
        stop_reason = property(GetStopReason, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eStopReason") that represents the reason this thread stopped.''')
        is_suspended = property(IsSuspended, None, doc='''A read only property that returns a boolean value that indicates if this thread is suspended.''')
        is_stopped = property(IsStopped, None, doc='''A read only property that returns a boolean value that indicates if this thread is stopped but not exited.''')
    %}
#endif

};

} // namespace lldb
