//===-- ThreadPlanSingleThreadTimeout.h -------------------------------------*-
// C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO
#ifndef LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H
#define LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/State.h"

#include <thread>

namespace lldb_private {

enum class SingleThreadPlanTimeoutState {
  InitialResume,
  TimeoutHalt,
  ResumingAllThreads,
  AfterThreadResumed,
};

class ThreadPlanSingleThreadTimeout : public ThreadPlan {
public:
  ThreadPlanSingleThreadTimeout(Thread &thread)
      : ThreadPlan(ThreadPlan::eKindSingleThreadTimeout,
                   "Single thread timeout", thread, eVoteNo, eVoteNoOpinion),
        m_state(SingleThreadPlanTimeoutState::InitialResume) {
    std::thread t(thread_function, this);
    t.detach();
  }

  ~ThreadPlanSingleThreadTimeout() override = default;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override {
    s->Printf("SingleThreadPlanTimeout, state(%d)", (int)m_state);
  }
  bool ValidatePlan(Stream *error) override { return true; }
  bool WillStop() override { return true; }
  bool DoPlanExplainsStop(Event *event_ptr) override { return true; }
  lldb::StateType GetPlanRunState() override { return lldb::eStateRunning; }
  static void thread_function(ThreadPlanSingleThreadTimeout *self) {
    int timeout_ms = 5000; // 5 seconds timeout
    std::this_thread::sleep_for(
        std::chrono::milliseconds(timeout_ms));
    self->HandleTimeout();
  }

  bool MischiefManaged() override {
    // return m_state == SingleThreadPlanTimeoutState::AfterThreadResumed;
    return GetPreviousPlan()->MischiefManaged();
  }

  bool ShouldStop(Event *event_ptr) override {
    if (m_state == SingleThreadPlanTimeoutState::InitialResume) {
      return GetPreviousPlan()->ShouldStop(event_ptr);
    }
    return HandleEvent(event_ptr);
  }

  void SetStopOthers(bool new_value) override {
    GetPreviousPlan()->SetStopOthers(new_value);
  }

  bool StopOthers() override {
    if (m_state == SingleThreadPlanTimeoutState::ResumingAllThreads ||
        m_state == SingleThreadPlanTimeoutState::AfterThreadResumed)
      return false;
    else
      return GetPreviousPlan()->StopOthers();
  }

protected:
  bool DoWillResume(lldb::StateType resume_state, bool current_plan) override {
    if (m_state == SingleThreadPlanTimeoutState::ResumingAllThreads) {
      m_state = SingleThreadPlanTimeoutState::AfterThreadResumed;
    }
    return GetPreviousPlan()->WillResume(resume_state, current_plan);
  }

  bool HandleEvent(Event *event_ptr) {
    lldb::StateType stop_state =
        Process::ProcessEventData::GetStateFromEvent(event_ptr);
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::HandleEvent(): got event: %s.",
              StateAsCString(stop_state));

    bool should_stop = true;
    if (m_state == SingleThreadPlanTimeoutState::TimeoutHalt &&
        stop_state == lldb::eStateStopped) {
      if (Process::ProcessEventData::GetRestartedFromEvent(event_ptr)) {
        // If we were restarted, we just need to go back up to fetch
        // another event.
        LLDB_LOGF(
            log, "ThreadPlanSingleThreadTimeout::HandleEvent(): Got a stop and "
                 "restart, so we'll continue waiting.");

      } else {
        GetThread().GetCurrentPlan()->SetStopOthers(false);
        m_state = SingleThreadPlanTimeoutState::ResumingAllThreads;
      }
      should_stop = false;
    }
    if (should_stop)
      return GetPreviousPlan()->ShouldStop(event_ptr);
    else
      return false;
  }

  void HandleTimeout() {
    m_state = SingleThreadPlanTimeoutState::TimeoutHalt;
    m_process.SendAsyncInterrupt();
  }

private:
  SingleThreadPlanTimeoutState m_state;

  ThreadPlanSingleThreadTimeout(const ThreadPlanSingleThreadTimeout &) = delete;
  const ThreadPlanSingleThreadTimeout &
  operator=(const ThreadPlanSingleThreadTimeout &) = delete;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H
