//===-- ThreadPlanStepOut.h -------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepOut_h_
#define liblldb_ThreadPlanStepOut_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanStepOut : public ThreadPlan
{
public:
    ThreadPlanStepOut (Thread &thread,
                       SymbolContext *addr_context,
                       bool first_insn,
                       bool stop_others,
                       Vote stop_vote,
                       Vote run_vote,
                       uint32_t frame_idx);

    virtual ~ThreadPlanStepOut ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();
    virtual void DidPush();
    virtual bool IsPlanStale();
    
    virtual lldb::ValueObjectSP GetReturnValueObject()
    {
        return m_return_valobj_sp;
    }

protected:
    virtual bool DoPlanExplainsStop (Event *event_ptr);
    virtual bool DoWillResume (lldb::StateType resume_state, bool current_plan);
    bool QueueInlinedStepPlan (bool queue_now);

private:
    SymbolContext *m_step_from_context;
    lldb::addr_t m_step_from_insn;
    StackID  m_step_out_to_id;
    StackID  m_immediate_step_from_id;
    lldb::break_id_t m_return_bp_id;
    lldb::addr_t m_return_addr;
    bool m_first_insn;
    bool m_stop_others;
    lldb::ThreadPlanSP m_step_through_inline_plan_sp;
    lldb::ThreadPlanSP m_step_out_plan_sp;
    Function          *m_immediate_step_from_function;
    lldb::ValueObjectSP m_return_valobj_sp;

    friend ThreadPlan *
    Thread::QueueThreadPlanForStepOut (bool abort_other_plans,
                                       SymbolContext *addr_context,
                                       bool first_insn,
                                       bool stop_others,
                                       Vote stop_vote,
                                       Vote run_vote,
                                       uint32_t frame_idx);

    // Need an appropriate marker for the current stack so we can tell step out
    // from step in.

    void
    CalculateReturnValue();
    
    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepOut);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepOut_h_
