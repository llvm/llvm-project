//===-- StopInfo.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_StopInfo_h_
#define liblldb_StopInfo_h_

// C Includes
// C++ Includes
#include <string>

// Other libraries and framework includes
// Project includes
#include "lldb/lldb-public.h"
#include "lldb/Target/Process.h"

namespace lldb_private {

class StopInfo
{
    friend Process::ProcessEventData;
    friend class ThreadPlanBase;
    
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    StopInfo (Thread &thread, uint64_t value);

    virtual ~StopInfo()
    {
    }


    bool
    IsValid () const;

    Thread &
    GetThread()
    {
        return m_thread;
    }

    const Thread &
    GetThread() const
    {
        return m_thread;
    }

    // The value of the StopInfo depends on the StopReason.
    // StopReason                  Meaning
    // ----------------------------------------------
    // eStopReasonBreakpoint       BreakpointSiteID
    // eStopReasonSignal           Signal number
    // eStopReasonWatchpoint       WatchpointLocationID
    // eStopReasonPlanComplete     No significance
    
    uint64_t
    GetValue() const
    {
        return m_value;
    }

    virtual lldb::StopReason
    GetStopReason () const = 0;
        
    // ShouldStopSynchronous will get called before any thread plans are consulted, and if it says we should
    // resume the target, then we will just immediately resume.  This should not run any code in or resume the
    // target.
    
    virtual bool
    ShouldStopSynchronous (Event *event_ptr)
    {
        return true;
    }

    // If should stop returns false, check if we should notify of this event
    virtual bool
    ShouldNotify (Event *event_ptr)
    {
        return false;
    }

    virtual void
    WillResume (lldb::StateType resume_state)
    {
        // By default, don't do anything
    }
    
    virtual const char *
    GetDescription ()
    {
        return m_description.c_str();
    }

    virtual void
    SetDescription (const char *desc_cstr)
    {
        if (desc_cstr && desc_cstr[0])
            m_description.assign (desc_cstr);
        else
            m_description.clear();
    }

    static lldb::StopInfoSP
    CreateStopReasonWithBreakpointSiteID (Thread &thread, lldb::break_id_t break_id);

    // This creates a StopInfo for the thread where the should_stop is already set, and won't be recalculated.
    static lldb::StopInfoSP
    CreateStopReasonWithBreakpointSiteID (Thread &thread, lldb::break_id_t break_id, bool should_stop);

    static lldb::StopInfoSP
    CreateStopReasonWithWatchpointID (Thread &thread, lldb::break_id_t watch_id);

    static lldb::StopInfoSP
    CreateStopReasonWithSignal (Thread &thread, int signo);

    static lldb::StopInfoSP
    CreateStopReasonToTrace (Thread &thread);

    static lldb::StopInfoSP
    CreateStopReasonWithPlan (lldb::ThreadPlanSP &plan, lldb::ValueObjectSP return_valobj_sp);

    static lldb::StopInfoSP
    CreateStopReasonWithException (Thread &thread, const char *description);
    
    static lldb::ValueObjectSP
    GetReturnValueObject (lldb::StopInfoSP &stop_info_sp);

protected:
    // Perform any action that is associated with this stop.  This is done as the
    // Event is removed from the event queue.  ProcessEventData::DoOnRemoval does the job.
    virtual void
    PerformAction (Event *event_ptr)
    {
    }

    // Stop the thread by default. Subclasses can override this to allow
    // the thread to continue if desired.  The ShouldStop method should not do anything
    // that might run code.  If you need to run code when deciding whether to stop
    // at this StopInfo, that must be done in the PerformAction.
    // The PerformAction will always get called before the ShouldStop.  This is done by the
    // ProcessEventData::DoOnRemoval, though the ThreadPlanBase needs to consult this later on.
    virtual bool
    ShouldStop (Event *event_ptr)
    {
        return true;
    }
    
    //------------------------------------------------------------------
    // Classes that inherit from StackID can see and modify these
    //------------------------------------------------------------------
    Thread &        m_thread;   // The thread corresponding to the stop reason.
    uint32_t        m_stop_id;  // The process stop ID for which this stop info is valid
    uint32_t        m_resume_id; // This is the resume ID when we made this stop ID.
    uint64_t        m_value;    // A generic value that can be used for things pertaining to this stop info
    std::string     m_description; // A textual description describing this stop.
    
    // This determines whether the target has run since this stop info.
    // N.B. running to evaluate a user expression does not count. 
    bool HasTargetRunSinceMe ();

    // MakeStopInfoValid is necessary to allow saved stop infos to resurrect themselves as valid.
    // It should only be used by Thread::RestoreThreadStateFromCheckpoint and to make sure the one-step
    // needed for before-the-fact watchpoints does not prevent us from stopping
    void
    MakeStopInfoValid ();
    
private:
    friend class Thread;
    
    DISALLOW_COPY_AND_ASSIGN (StopInfo);
};

} // namespace lldb_private

#endif  // liblldb_StopInfo_h_
