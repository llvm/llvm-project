//===-- TimeoutResumeAll.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_TIMEOUTRESUMEALL_H
#define LLDB_TARGET_TIMEOUTRESUMEALL_H

#include "lldb/Target/ThreadPlanSingleThreadTimeout.h"

namespace lldb_private {

// Mixin class that provides the capability for ThreadPlan to support single
// thread execution that resumes all threads after a timeout.
// Opt-in thread plan should call PushNewTimeout() in its DidPush() and
// ResumeWithTimeout() during DoWillResume().
class TimeoutResumeAll {
public:
  TimeoutResumeAll(Thread &thread)
      : m_thread(thread),
        m_timeout_info(
            std::make_shared<ThreadPlanSingleThreadTimeout::TimeoutInfo>()) {}

  void PushNewTimeout() {
    ThreadPlanSingleThreadTimeout::PushNewWithTimeout(m_thread, m_timeout_info);
  }

  void ResumeWithTimeout() {
    ThreadPlanSingleThreadTimeout::ResumeFromPrevState(m_thread,
                                                       m_timeout_info);
  }

private:
  Thread &m_thread;
  ThreadPlanSingleThreadTimeout::TimeoutInfoSP m_timeout_info;
};

} // namespace lldb_private

#endif // LLDB_TARGET_TIMEOUTRESUMEALL_H
