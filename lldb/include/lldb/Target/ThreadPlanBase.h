//===-- ThreadPlanBase.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanFundamental_h_
#define liblldb_ThreadPlanFundamental_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Process.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {


//------------------------------------------------------------------
//  Base thread plans:
//  This is the generic version of the bottom most plan on the plan stack.  It should
//  be able to handle generic breakpoint hitting, and signals and exceptions.
//------------------------------------------------------------------

class ThreadPlanBase : public ThreadPlan
{
friend class Process;  // RunThreadPlan manages "stopper" base plans.
public:
    virtual ~ThreadPlanBase ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool ShouldStop (Event *event_ptr);
    virtual Vote ShouldReportStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

    virtual bool OkayToDiscard()
    {
        return false;
    }

    virtual bool
    IsBasePlan()
    {
        return true;
    }
    
protected:
    virtual bool DoWillResume (lldb::StateType resume_state, bool current_plan);
    virtual bool DoPlanExplainsStop (Event *event_ptr);
    ThreadPlanBase (Thread &thread);

private:
    friend ThreadPlan *
    Thread::QueueFundamentalPlan(bool abort_other_plans);

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanBase);
};


} // namespace lldb_private

#endif  // liblldb_ThreadPlanFundamental_h_
