//===-- ThreadPlanStepOverBreakpoint.h --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepOverBreakpoint_h_
#define liblldb_ThreadPlanStepOverBreakpoint_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanStepOverBreakpoint : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepOverBreakpoint ();

    ThreadPlanStepOverBreakpoint (Thread &thread);
    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool PlanExplainsStop ();
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillResume (lldb::StateType resume_state, bool current_plan);
    virtual bool WillStop ();
    virtual bool MischiefManaged ();
    void SetAutoContinue (bool do_it);
    virtual bool ShouldAutoContinue(Event *event_ptr);

protected:

private:

    lldb::addr_t m_breakpoint_addr;
    lldb::user_id_t m_breakpoint_site_id;
    bool m_auto_continue;

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepOverBreakpoint);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepOverBreakpoint_h_
