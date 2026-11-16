//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADPLANRUNTOBREAKPOINT_H
#define LLDB_TARGET_THREADPLANRUNTOBREAKPOINT_H

#include <vector>

#include "lldb/Target/ThreadPlan.h"
#include "lldb/lldb-private.h"

namespace lldb_private {

// This plan runs till it hits one of the breakpoints passed in.
// It takes ownership of the breakpoint, and will delete it when
// the plan is complete.
// ThreadPlanRunToAddress is more effient if you know the target address up
// front.  Use this one when the function you want to stop at might be realized
// in the course of the thread plan operation, so you need a breakpoint that
// can update itself.

class ThreadPlanRunToBreakpoint : public ThreadPlan {
public:
  ThreadPlanRunToBreakpoint(Thread &thread, lldb::BreakpointSP breakpoint_sp,
                            bool stop_others);

  ThreadPlanRunToBreakpoint(Thread &thread,
                            const std::vector<lldb::BreakpointSP> &breakpoints,
                            bool stop_others);

  ~ThreadPlanRunToBreakpoint() override;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;

  bool ValidatePlan(Stream *error) override;

  bool ShouldStop(Event *event_ptr) override;

  bool StopOthers() override;

  void SetStopOthers(bool new_value) override;

  lldb::StateType GetPlanRunState() override;

  bool WillStop() override;

  bool MischiefManaged() override;

protected:
  bool DoPlanExplainsStop(Event *event_ptr) override;

  bool AtOurBreakpoint();

private:
  bool m_stop_others;
  std::vector<lldb::BreakpointSP>
      m_breakpoints; // This is the list of breakpoints we are going to run to.

  ThreadPlanRunToBreakpoint(const ThreadPlanRunToBreakpoint &) = delete;
  const ThreadPlanRunToBreakpoint &
  operator=(const ThreadPlanRunToBreakpoint &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANRUNTOBREAKPOINT_H
