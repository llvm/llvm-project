//===-- ThreadPlanSingleThreadTimeout.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H
#define LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H

#include "lldb/Target/Thread.h"
#include "lldb/Target/ThreadPlan.h"
#include "lldb/Utility/Event.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/State.h"

#include <chrono>
#include <thread>

namespace lldb_private {

class ThreadPlanSingleThreadTimeout;
//
// Thread plan used by single thread execution to issue timeout. This is useful
// to detect potential deadlock in single thread execution. The timeout measures
// the elapsed time from the last internal stop and gets reset by each internal
// stop to ensure we are accurately detecting execution not moving forward.
// This means this thread plan may be created/destroyed multiple times by the
// parent execution plan.
//
// When a timeout happens, the thread plan resolves the potential deadlock by
// issuing a thread specific async interrupt to enter stop state, then execution
// is resumed with all threads running to resolve the potential deadlock
//
class ThreadPlanSingleThreadTimeout : public ThreadPlan {
  enum class State {
    WaitTimeout,    // Waiting for timeout.
    AsyncInterrupt, // Async interrupt has been issued.
    Done,           // Finished resume all threads.
  };

public:
  // TODO: allow timeout to be set on per thread plan basis.
  struct TimeoutInfo {
    // Whether there is a ThreadPlanSingleThreadTimeout instance alive.
    bool m_isAlive = false;
    ThreadPlanSingleThreadTimeout::State m_last_state = State::WaitTimeout;
  };
  using TimeoutInfoSP =
      std::shared_ptr<ThreadPlanSingleThreadTimeout::TimeoutInfo>;

  ~ThreadPlanSingleThreadTimeout() override;

  // If input \param thread is running in single thread mode, push a
  // new ThreadPlanSingleThreadTimeout based on timeout setting from fresh new
  // state. The reference of \param info is passed in so that when
  // ThreadPlanSingleThreadTimeout got popped its last state can be stored
  // in it for future resume.
  static void PushNewWithTimeout(Thread &thread, TimeoutInfoSP &info);

  // Push a new ThreadPlanSingleThreadTimeout by restoring state from
  // input \param info and resume execution.
  static void ResumeFromPrevState(Thread &thread, TimeoutInfoSP &info);

  void GetDescription(Stream *s, lldb::DescriptionLevel level) override;
  bool ValidatePlan(Stream *error) override { return true; }
  bool WillStop() override;
  void DidPop() override;

  bool IsLeafPlan() override { return true; }
  bool DoPlanExplainsStop(Event *event_ptr) override;

  lldb::StateType GetPlanRunState() override;
  static void TimeoutThreadFunc(ThreadPlanSingleThreadTimeout *self);

  bool MischiefManaged() override;

  bool ShouldStop(Event *event_ptr) override;
  void SetStopOthers(bool new_value) override;
  bool StopOthers() override;

private:
  ThreadPlanSingleThreadTimeout(Thread &thread, TimeoutInfoSP &info);

  bool IsTimeoutAsyncInterrupt(Event *event_ptr);
  bool HandleEvent(Event *event_ptr);
  void HandleTimeout();
  uint64_t GetRemainingTimeoutMilliSeconds();

  static std::string StateToString(State state);

  ThreadPlanSingleThreadTimeout(const ThreadPlanSingleThreadTimeout &) = delete;
  const ThreadPlanSingleThreadTimeout &
  operator=(const ThreadPlanSingleThreadTimeout &) = delete;

  TimeoutInfoSP m_info; // Reference to controlling ThreadPlan's TimeoutInfo.
  State m_state;

  // Lock for m_wakeup_cv and m_exit_flag between thread plan thread and timer
  // thread
  std::mutex m_mutex;
  std::condition_variable m_wakeup_cv;
  std::thread m_timer_thread;
  std::chrono::steady_clock::time_point m_timeout_start;
};

} // namespace lldb_private

#endif // LLDB_TARGET_THREADPLANSINGLETHREADTIMEOUT_H
