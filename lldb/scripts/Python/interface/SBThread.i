//===-- SWIG Interface for SBThread -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"Represents a thread of execution. SBProcess contains SBThread(s).

SBThreads can be referred to by their ID, which maps to the system specific thread
identifier, or by IndexID.  The ID may or may not be unique depending on whether the
system reuses its thread identifiers.  The IndexID is a monotonically increasing identifier
that will always uniquely reference a particular thread, and when that thread goes
away it will not be reused.

SBThread supports frame iteration. For example (from test/python_api/
lldbutil/iter/TestLLDBIterator.py),

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

See also SBProcess and SBFrame."
) SBThread;
class SBThread
{
public:
    SBThread ();

    SBThread (const lldb::SBThread &thread);

   ~SBThread();
    
    static bool
    EventIsThreadEvent (const SBEvent &event);
    
    static SBFrame
    GetStackFrameFromEvent (const SBEvent &event);
    
    static SBThread
    GetThreadFromEvent (const SBEvent &event);

    bool
    IsValid() const;

    void
    Clear ();

    lldb::StopReason
    GetStopReason();

    %feature("docstring", "
    /// Get the number of words associated with the stop reason.
    /// See also GetStopReasonDataAtIndex().
    ") GetStopReasonDataCount;
    size_t
    GetStopReasonDataCount();

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// Get information associated with a stop reason.
    ///
    /// Breakpoint stop reasons will have data that consists of pairs of 
    /// breakpoint IDs followed by the breakpoint location IDs (they always come
    /// in pairs).
    ///
    /// Stop Reason              Count Data Type
    /// ======================== ===== =========================================
    /// eStopReasonNone          0
    /// eStopReasonTrace         0
    /// eStopReasonBreakpoint    N     duple: {breakpoint id, location id}
    /// eStopReasonWatchpoint    1     watchpoint id
    /// eStopReasonSignal        1     unix signal number
    /// eStopReasonException     N     exception data
    /// eStopReasonPlanComplete  0
    //--------------------------------------------------------------------------
    ") GetStopReasonDataAtIndex;
    uint64_t
    GetStopReasonDataAtIndex(uint32_t idx);

    %feature("autodoc", "
    Pass only an (int)length and expect to get a Python string describing the
    stop reason.
    ") GetStopDescription;
    size_t
    GetStopDescription (char *dst, size_t dst_len);

    SBValue
    GetStopReturnValue ();

    lldb::tid_t
    GetThreadID () const;

    uint32_t
    GetIndexID () const;

    const char *
    GetName () const;

    const char *
    GetQueueName() const;

    void
    StepOver (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepInto (lldb::RunMode stop_other_threads = lldb::eOnlyDuringStepping);

    void
    StepOut ();

    void
    StepOutOfFrame (lldb::SBFrame &frame);

    void
    StepInstruction(bool step_over);

    SBError
    StepOverUntil (lldb::SBFrame &frame, 
                   lldb::SBFileSpec &file_spec, 
                   uint32_t line);

    void
    RunToAddress (lldb::addr_t addr);

    SBError
    ReturnFromFrame (SBFrame &frame, SBValue &return_value);

    %feature("docstring", "
    //--------------------------------------------------------------------------
    /// LLDB currently supports process centric debugging which means when any
    /// thread in a process stops, all other threads are stopped. The Suspend()
    /// call here tells our process to suspend a thread and not let it run when
    /// the other threads in a process are allowed to run. So when 
    /// SBProcess::Continue() is called, any threads that aren't suspended will
    /// be allowed to run. If any of the SBThread functions for stepping are 
    /// called (StepOver, StepInto, StepOut, StepInstruction, RunToAddres), the
    /// thread will now be allowed to run and these funtions will simply return.
    ///
    /// Eventually we plan to add support for thread centric debugging where
    /// each thread is controlled individually and each thread would broadcast
    /// its state, but we haven't implemented this yet.
    /// 
    /// Likewise the SBThread::Resume() call will again allow the thread to run
    /// when the process is continued.
    ///
    /// Suspend() and Resume() functions are not currently reference counted, if
    /// anyone has the need for them to be reference counted, please let us
    /// know.
    //--------------------------------------------------------------------------
    ") Suspend;
    bool
    Suspend();
    
    bool
    Resume ();
    
    bool
    IsSuspended();

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
    
    bool
    GetStatus (lldb::SBStream &status) const;
    
    %pythoncode %{
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
        
        __swig_getmethods__["id"] = GetThreadID
        if _newclass: id = property(GetThreadID, None, doc='''A read only property that returns the thread ID as an integer.''')

        __swig_getmethods__["idx"] = GetIndexID
        if _newclass: idx = property(GetIndexID, None, doc='''A read only property that returns the thread index ID as an integer. Thread index ID values start at 1 and increment as threads come and go and can be used to uniquely identify threads.''')

        __swig_getmethods__["return_value"] = GetStopReturnValue
        if _newclass: return_value = property(GetStopReturnValue, None, doc='''A read only property that returns an lldb object that represents the return value from the last stop (lldb.SBValue) if we just stopped due to stepping out of a function.''')

        __swig_getmethods__["process"] = GetProcess
        if _newclass: process = property(GetProcess, None, doc='''A read only property that returns an lldb object that represents the process (lldb.SBProcess) that owns this thread.''')

        __swig_getmethods__["num_frames"] = GetNumFrames
        if _newclass: num_frames = property(GetNumFrames, None, doc='''A read only property that returns the number of stack frames in this thread as an integer.''')

        __swig_getmethods__["frames"] = get_thread_frames
        if _newclass: frames = property(get_thread_frames, None, doc='''A read only property that returns a list() of lldb.SBFrame objects for all frames in this thread.''')

        __swig_getmethods__["frame"] = get_frames_access_object
        if _newclass: frame = property(get_frames_access_object, None, doc='''A read only property that returns an object that can be used to access frames as an array ("frame_12 = lldb.thread.frame[12]").''')

        __swig_getmethods__["name"] = GetName
        if _newclass: name = property(GetName, None, doc='''A read only property that returns the name of this thread as a string.''')

        __swig_getmethods__["queue"] = GetQueueName
        if _newclass: queue = property(GetQueueName, None, doc='''A read only property that returns the dispatch queue name of this thread as a string.''')

        __swig_getmethods__["stop_reason"] = GetStopReason
        if _newclass: stop_reason = property(GetStopReason, None, doc='''A read only property that returns an lldb enumeration value (see enumerations that start with "lldb.eStopReason") that represents the reason this thread stopped.''')

        __swig_getmethods__["is_suspended"] = IsSuspended
        if _newclass: is_suspended = property(IsSuspended, None, doc='''A read only property that returns a boolean value that indicates if this thread is suspended.''')
    %}

};

} // namespace lldb
