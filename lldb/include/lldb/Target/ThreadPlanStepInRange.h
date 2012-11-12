//===-- ThreadPlanStepInRange.h ---------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepInRange_h_
#define liblldb_ThreadPlanStepInRange_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/Core/AddressRange.h"
#include "lldb/Target/StackID.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepRange.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"

namespace lldb_private {

class ThreadPlanStepInRange :
    public ThreadPlanStepRange,
    public ThreadPlanShouldStopHere
{
public:
    ThreadPlanStepInRange (Thread &thread,
                           const AddressRange &range,
                           const SymbolContext &addr_context,
                           lldb::RunMode stop_others);

    virtual
    ~ThreadPlanStepInRange ();

    virtual void
    GetDescription (Stream *s, lldb::DescriptionLevel level);

    virtual bool
    ShouldStop (Event *event_ptr);

    void SetAvoidRegexp(const char *name);
    
    static ThreadPlan *
    DefaultShouldStopHereCallback (ThreadPlan *current_plan, Flags &flags, void *baton);

    static void
    SetDefaultFlagValue (uint32_t new_value);

    virtual bool
    PlanExplainsStop ();

    virtual bool WillResume (lldb::StateType resume_state, bool current_plan);
    
protected:

    virtual void
    SetFlagsToDefault ();
    
    bool
    FrameMatchesAvoidRegexp ();

private:

    friend ThreadPlan *
    Thread::QueueThreadPlanForStepRange (bool abort_other_plans,
                                         StepType type,
                                         const AddressRange &range,
                                         const SymbolContext &addr_context,
                                         lldb::RunMode stop_others,
                                         bool avoid_code_without_debug_info);


    // Need an appropriate marker for the current stack so we can tell step out
    // from step in.

    static uint32_t s_default_flag_values;
    std::auto_ptr<RegularExpression> m_avoid_regexp_ap;
    bool m_step_past_prologue;  // FIXME: For now hard-coded to true, we could put a switch in for this if there's
                                // demand for that.
    bool m_virtual_step;        // true if we've just done a "virtual step", i.e. just moved the inline stack depth.

    DISALLOW_COPY_AND_ASSIGN (ThreadPlanStepInRange);

};

} // namespace lldb_private

#endif  // liblldb_ThreadPlanStepInRange_h_
