//===-- ThreadPlanStepUntil.h -----------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepUntil_h_
#define liblldb_ThreadPlanStepUntil_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {


class ThreadPlanStepUntil : public ThreadPlan
{
public:
    virtual ~ThreadPlanStepUntil ();

    virtual void GetDescription (Stream *s, lldb::DescriptionLevel level);
    virtual bool ValidatePlan (Stream *error);
    virtual bool ShouldStop (Event *event_ptr);
    virtual bool StopOthers ();
    virtual lldb::StateType GetPlanRunState ();
    virtual bool WillStop ();
    virtual bool MischiefManaged ();

protected:
    virtual bool DoWillResume (lldb::StateType resume_state, bool current_plan);
    virtual bool DoPlanExplainsStop (Event *event_ptr);

    ThreadPlanStepUntil (Thread &thread,
                         lldb::addr_t *address_list,
                         size_t num_addresses,
                         bool stop_others,
                         uint32_t frame_idx = 0);
    void AnalyzeStop(void);

private:

    StackID m_stack_id;
    lldb::addr_t m_step_from_insn;
    lldb::break_id_t m_return_bp_id;
    lldb::addr_t m_return_addr;
    bool m_stepped_out;
    bool m_should_stop;
    bool m_ran_analyze;
    bool m_explains_stop;

    typedef std::map<lldb::addr_t,lldb::break_id_t> until_collection;
    until_collection m_until_points;
    bool m_stop_others;

    void Clear();

    friend ThreadPlan *
    Thread::QueueThreadPlanForStepUntil (bool abort_other_plans,
                                         lldb::addr_t *address_list,
                                         size_t num_addresses,
                                         bool stop_others,
                                         uint32_t frame_idx);

    // Need an appropriate marker for the current stack so we can tell step out
    // from step in.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepUntil);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepUntil_h_
