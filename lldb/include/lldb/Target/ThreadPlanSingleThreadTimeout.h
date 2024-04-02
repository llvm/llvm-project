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
  WaitTimeout,
  AsyncInterrupt,
  Done,
};

class ThreadPlanSingleThreadTimeout : public ThreadPlan {
public:
  ThreadPlanSingleThreadTimeout(Thread &thread)
      : ThreadPlan(ThreadPlan::eKindSingleThreadTimeout,
                   "Single thread timeout", thread, eVoteNo, eVoteNoOpinion),
        m_state(SingleThreadPlanTimeoutState::WaitTimeout) {
    m_timer_thread = std::thread(thread_function, this);
  }

  ~ThreadPlanSingleThreadTimeout() override = default;

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override {
    s->Printf("SingleThreadPlanTimeout, state(%d)", (int)m_state);
  }
  bool ValidatePlan(Stream *error) override { return true; }
  bool WillStop() override { return true; }

  void DidPop() override {
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::DidPop().");
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      // Tell timer thread to exit.
      m_exit_flag = true;
    }
    // Wait for timer thread to exit.
    m_timer_thread.join();
  }

  bool DoPlanExplainsStop(Event *event_ptr) override { 
    lldb::StateType stop_state =
        Process::ProcessEventData::GetStateFromEvent(event_ptr);
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::DoPlanExplainsStop(): got event: %s.",
              StateAsCString(stop_state));
    lldb::StopInfoSP stop_info = GetThread().GetStopInfo();
    return m_state == SingleThreadPlanTimeoutState::AsyncInterrupt &&
        stop_state == lldb::eStateStopped && stop_info &&
        stop_info->GetStopReason() == lldb::eStopReasonInterrupt;
  }

  lldb::StateType GetPlanRunState() override { return lldb::eStateStepping; }
  static void thread_function(ThreadPlanSingleThreadTimeout *self) {
    std::unique_lock<std::mutex> lock(self->m_mutex);
    self->m_wakeup_cv.wait_for(lock, std::chrono::seconds(1));

    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::thread_function() called with m_exit_flag(%d).", self->m_exit_flag);
    if (self->m_exit_flag)
      return;

    self->HandleTimeout();
  }

  bool MischiefManaged() override {
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::MischiefManaged() called.");
    // Should reset timer on each internal stop/execution progress.
    return true;
  }

  bool ShouldStop(Event *event_ptr) override {
    if (m_state == SingleThreadPlanTimeoutState::WaitTimeout) {
      return GetPreviousPlan()->ShouldStop(event_ptr);
    }
    return HandleEvent(event_ptr);
  }

  void SetStopOthers(bool new_value) override {
    GetPreviousPlan()->SetStopOthers(new_value);
  }

  bool StopOthers() override {
    if (m_state == SingleThreadPlanTimeoutState::Done)
      return false;
    else
      return GetPreviousPlan()->StopOthers();
  }

protected:
  // bool DoWillResume(lldb::StateType resume_state, bool current_plan) override {
  //   if (m_state == SingleThreadPlanTimeoutState::Done) {
  //     m_state = SingleThreadPlanTimeoutState::AfterThreadResumed;
  //   }
  //   return GetPreviousPlan()->WillResume(resume_state, current_plan);
  // }

  bool HandleEvent(Event *event_ptr) {
    lldb::StateType stop_state =
        Process::ProcessEventData::GetStateFromEvent(event_ptr);
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::HandleEvent(): got event: %s.",
              StateAsCString(stop_state));

    lldb::StopInfoSP stop_info = GetThread().GetStopInfo();
    if (m_state == SingleThreadPlanTimeoutState::AsyncInterrupt &&
        stop_state == lldb::eStateStopped && stop_info &&
        stop_info->GetStopReason() == lldb::eStopReasonInterrupt) {
      if (Process::ProcessEventData::GetRestartedFromEvent(event_ptr)) {
        // If we were restarted, we just need to go back up to fetch
        // another event.
        LLDB_LOGF(
            log, "ThreadPlanSingleThreadTimeout::HandleEvent(): Got a stop and "
                 "restart, so we'll continue waiting.");

      } else {
        LLDB_LOGF(
            log, "ThreadPlanSingleThreadTimeout::HandleEvent(): Got async interrupt "
                 ", so we will resume all threads.");
        GetThread().SetStopOthers(false);
        m_state = SingleThreadPlanTimeoutState::Done;
      }
    }
    // Should not report stop.
    return false;
  }

  void HandleTimeout() {
    Log *log = GetLog(LLDBLog::Step);
    LLDB_LOGF(log,
              "ThreadPlanSingleThreadTimeout::HandleTimeout() send async interrupt.");
    // TODO: mutex
    m_state = SingleThreadPlanTimeoutState::AsyncInterrupt;

    // Private state thread will only send async interrupt
    // in running state so no need to check state here.
    m_process.SendAsyncInterrupt(&GetThread());
  }

private:
  SingleThreadPlanTimeoutState m_state;

  ThreadPlanSingleThreadTimeout(const ThreadPlanSingleThreadTimeout &) = delete;
  const ThreadPlanSingleThreadTimeout &
  operator=(const ThreadPlanSingleThreadTimeout &) = delete;

  std::mutex m_mutex;
  std::condition_variable m_wakeup_cv;
  bool m_exit_flag = false;
  std::thread m_timer_thread;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H
