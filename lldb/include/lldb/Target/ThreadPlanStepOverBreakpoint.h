//===-- ThreadPlanStepOverBreakpoint.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADPLANSTEPOVERBREAKPOINT_H
#define LLDB_TARGET_THREADPLANSTEPOVERBREAKPOINT_H

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"

namespace lldb_private {

class ThreadPlanStepOverBreakpoint : public ThreadPlan {
public:
  ThreadPlanStepOverBreakpoint(Thread &thread);

  ~ThreadPlanStepOverBreakpoint() override;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;
  bool ValidatePlan(Stream *error) override;
  bool ShouldStop(Event *event_ptr) override;
  bool SupportsResumeOthers() override;
  bool StopOthers() override;
  lldb::StateType GetPlanRunState() override;
  bool WillStop() override;
  void DidPop() override;
  bool MischiefManaged() override;
  void ThreadDestroyed() override;
  void SetAutoContinue(bool do_it);
  bool ShouldAutoContinue(Event *event_ptr) override;
  bool IsPlanStale() override;

  lldb::addr_t GetBreakpointLoadAddress() const { return m_breakpoint_addr; }

  /// When set to true, the breakpoint site will NOT be re-enabled directly
  /// by this plan. Instead, the plan will call
  /// ThreadList::ThreadFinishedSteppingOverBreakpoint() when it completes,
  /// allowing ThreadList to track all threads stepping over the same
  /// breakpoint and only re-enable it when ALL threads have finished.
  void SetDeferReenableBreakpointSite(bool defer) {
    m_defer_reenable_breakpoint_site = defer;
  }

  bool GetDeferReenableBreakpointSite() const {
    return m_defer_reenable_breakpoint_site;
  }

  /// Mark the breakpoint site as already re-enabled, suppressing any
  /// re-enable in DidPop()/ThreadDestroyed(). Used when discarding plans
  /// during WillResume cleanup to avoid spurious breakpoint toggles.
  void SetReenabledBreakpointSite() { m_reenabled_breakpoint_site = true; }

protected:
  bool DoPlanExplainsStop(Event *event_ptr) override;
  bool DoWillResume(lldb::StateType resume_state, bool current_plan) override;

  void ReenableBreakpointSite();

private:
  lldb::addr_t m_breakpoint_addr;
  lldb::user_id_t m_breakpoint_site_id;
  bool m_auto_continue;
  bool m_reenabled_breakpoint_site;
  bool m_defer_reenable_breakpoint_site;

  ThreadPlanStepOverBreakpoint(const ThreadPlanStepOverBreakpoint &) = delete;
  const ThreadPlanStepOverBreakpoint &
  operator=(const ThreadPlanStepOverBreakpoint &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSTEPOVERBREAKPOINT_H
