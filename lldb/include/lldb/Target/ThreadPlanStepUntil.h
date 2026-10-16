//===-- ThreadPlanStepUntil.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADPLANSTEPUNTIL_H
#define LLDB_TARGET_THREADPLANSTEPUNTIL_H

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlanStepOut.h"

namespace lldb_private {

class ThreadPlanStepUntil : public ThreadPlanStepOut {
public:
  ThreadPlanStepUntil(Thread &thread, llvm::ArrayRef<lldb::addr_t> address_list,
                      bool stop_others, uint32_t frame_idx = 0);

  ~ThreadPlanStepUntil() override;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;
  bool ValidatePlan(Stream *error) override;
  bool ShouldStop(Event *event_ptr) override;
  bool WillStop() override;
  bool MischiefManaged() override;
  bool IsPlanStale() override;

protected:
  bool DoWillResume(lldb::StateType resume_state, bool current_plan) override;
  bool DoPlanExplainsStop(Event *event_ptr) override;

private:
  /// Returns the site of the stop if it holds an until point.
  lldb::BreakpointSiteSP GetUntilPointSite();
  /// Completes the plan if frame zero is the until frame.
  void CompleteIfInUntilFrame();

  StackID m_stack_id;
  lldb::addr_t m_step_from_insn;
  bool m_reached_until_point = false;

  typedef std::map<lldb::addr_t, lldb::break_id_t> until_collection;
  until_collection m_until_points;

  void Clear();
  void SetUntilPointsEnabled(bool enabled);

  ThreadPlanStepUntil(const ThreadPlanStepUntil &) = delete;
  const ThreadPlanStepUntil &operator=(const ThreadPlanStepUntil &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSTEPUNTIL_H
