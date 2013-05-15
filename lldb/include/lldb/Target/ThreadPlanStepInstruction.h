//===-- ThreadPlanStepInstruction.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepInstruction_h_
#define liblldb_ThreadPlanStepInstruction_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanStepInstruction : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepInstruction ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

protected:
    virtual bool DoPlanExplainsStop (Event *event_ptr);

    ThreadPlanStepInstruction (Thread &thread,
                               bool step_over,
                               bool stop_others,
                               Vote stop_vote,
                               Vote run_vote);

private:
    friend ThreadPlan *
    Thread::QueueThreadPlanForStepSingleInstruction (bool step_over, bool abort_other_plans, bool stop_other_threads);

    lldb::addr_t m_instruction_addr;
    bool m_stop_other_threads;
    bool m_step_over;
    // This is used only for the step over case.
    StackID m_stack_id;

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepInstruction);

};


} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepInstruction_h_
