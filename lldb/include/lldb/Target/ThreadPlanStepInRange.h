//===-- ThreadPlanStepInRange.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ThreadPlanStepInRange_h_
#define liblldb_ThreadPlanStepInRange_h_

#include "lldb/Core/AddressRange.h"
#include "lldb/Target/StackID.h"
#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanShouldStopHere.h"
#include "lldb/Target/ThreadPlanStepRange.h"

namespace lldb_private {

class ThreadPlanStepInRange : public ThreadPlanStepRange,
                              public ThreadPlanShouldStopHere {
public:
  ThreadPlanStepInRange(Thread &thread, const AddressRange &range,
                        const SymbolContext &addr_context,
                        lldb::RunMode stop_others,
                        LazyBool step_in_avoids_code_without_debug_info,
                        LazyBool step_out_avoids_code_without_debug_info);

  ThreadPlanStepInRange(Thread &thread, const AddressRange &range,
                        const SymbolContext &addr_context,
                        const char *step_into_function_name,
                        lldb::RunMode stop_others,
                        LazyBool step_in_avoids_code_without_debug_info,
                        LazyBool step_out_avoids_code_without_debug_info);

  ~ThreadPlanStepInRange() override;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;

  bool ShouldStop(Event *event_ptr) override;

  void SetAvoidRegexp(const char *name);

  void SetStepInTarget(const char *target) {
    m_step_into_target.SetCString(target);
  }

  static void SetDefaultFlagValue(uint32_t new_value);

  bool IsVirtualStep() override;

  bool MischiefManaged() override;

protected:
  static bool DefaultShouldStopHereCallback(ThreadPlan *current_plan,
                                            Flags &flags,
                                            lldb::FrameComparison operation,
                                            Status &status, void *baton);

  bool DoWillResume(lldb::StateType resume_state, bool current_plan) override;

  bool DoPlanExplainsStop(Event *event_ptr) override;

  void SetFlagsToDefault() override {
    GetFlags().Set(ThreadPlanStepInRange::s_default_flag_values);
  }

  void SetCallbacks() {
    ThreadPlanShouldStopHere::ThreadPlanShouldStopHereCallbacks callbacks(
        ThreadPlanStepInRange::DefaultShouldStopHereCallback, nullptr);
    SetShouldStopHereCallbacks(&callbacks, nullptr);
  }

  bool FrameMatchesAvoidCriteria();

private:
  friend lldb::ThreadPlanSP Thread::QueueThreadPlanForStepOverRange(
      bool abort_other_plans, const AddressRange &range,
      const SymbolContext &addr_context, lldb::RunMode stop_others,
      Status &status, LazyBool avoid_code_without_debug_info);
  friend lldb::ThreadPlanSP Thread::QueueThreadPlanForStepInRange(
      bool abort_other_plans, const AddressRange &range,
      const SymbolContext &addr_context, const char *step_in_target,
      lldb::RunMode stop_others, Status &status,
      LazyBool step_in_avoids_code_without_debug_info,
      LazyBool step_out_avoids_code_without_debug_info);

  void SetupAvoidNoDebug(LazyBool step_in_avoids_code_without_debug_info,
                         LazyBool step_out_avoids_code_without_debug_info);

  bool DefaultShouldStopHereImpl(Flags &flags, bool should_step_out);

  bool StepInDeepBreakpointExplainsStop(lldb::StopInfoSP stop_info_sp);

  void ClearStepInDeepBreakpoints();

  // Need an appropriate marker for the current stack so we can tell step out
  // from step in.

  static uint32_t s_default_flag_values; // These are the default flag values
                                         // for the ThreadPlanStepThrough.
  lldb::ThreadPlanSP m_sub_plan_sp;      // Keep track of the last plan we were
                                    // running.  If it fails, we should stop.
  std::unique_ptr<RegularExpression> m_avoid_regexp_ap;
  bool m_step_past_prologue; // FIXME: For now hard-coded to true, we could put
                             // a switch in for this if there's
                             // demand for that.
  bool m_virtual_step; // true if we've just done a "virtual step", i.e. just
                       // moved the inline stack depth.
  ConstString m_step_into_target;
  std::vector<lldb::break_id_t> m_step_in_deep_bps; // Places where we might
                                                    // want to stop when we do a
                                                    // step out.
  DISALLOW_COPY_AND_ASSIGN(ThreadPlanStepInRange);
};

} // namespace lldb_private

#endif // liblldb_ThreadPlanStepInRange_h_
