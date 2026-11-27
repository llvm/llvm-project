//===-- tsan_simulate.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
// Simulation scheduler for systematic thread interleaving exploration.
// Inspired by Relacy Race Detector's random scheduler:
// https://github.com/dvyukov/relacy
//
// When simulation is active, exactly one application thread runs at a time.
// Other threads are parked on internal semaphores. At each sync point
// (pthread_* calls, atomic operations), the running thread may yield to another
// thread chosen by the scheduler.
//===----------------------------------------------------------------------===//

#ifndef TSAN_SIMULATE_H
#define TSAN_SIMULATE_H

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __tsan {

// TODO: Simulation would be more useful with the following features
//  - Read/write mutex support
//  - Timed pthread* API support
//  - std::atomic::wait/notify* support (doesn't work today, because these APIs
//    rely on direct OS futex calls which TSAN does not observe)
//  - Alternate scheduling algorithms like full search or other random
//    distributions

// Run the simulation: invoke `callback(arg)` for `simulate_iterations`
// iterations, exploring thread interleavings using the configured scheduler.
// The callback runs only once if simulate_scheduler is empty (no simulation)
// or if the first iteration creates no threads.
// Returns 0 on success, -1 on error.
//
// Errors include
//  - Pre-existing threads when simulation was started
//  - Nested call while a simulation is running
//  - Unknown simulate_scheduler
//  - Invalid simulate_iterations, simulate_start_iteration, or
//    simulate_schedule_probability
//  - Race detected
//  - Thread leak (a thread finished but was never joined or detached)
//
// The following errors result in program termination via Die()
//  - Deadlock detected (all simulated threads were blocked)
//  - Unsupported interceptor called
//  - More than simulate_max_depth scheduling points in an iteration
//  - More than kMaxSimThreads threads, including the callback thread, in an
//    iteration
int SimulateRun(void (*callback)(void*), void* arg);

extern bool sim_active;

ALWAYS_INLINE bool SimulateIsActive() { return sim_active; }

void SimulateScheduleImpl();
NORETURN void SimulateReportUnsupportedImpl(const char* func_name);
void SimulateReportRaceImpl();
void SimulateThreadRegisterImpl(uptr thread_handle);
void SimulateBeforeChildThreadRunsImpl();
void SimulateThreadFinishImpl();

// SimulateSchedule is the key hook for simulation. It's called at each
// scheduling point (atomic op, mutex/cv op, thread create/join). When
// simulation is active, SimulateSchedule will check if another thread should
// run, and if so, context switch to that thread.
ALWAYS_INLINE void SimulateSchedule() {
  if (LIKELY(!SimulateIsActive()))
    return;
  SimulateScheduleImpl();
}

// Thread lifecycle

ALWAYS_INLINE void SimulateThreadRegister(uptr thread_handle) {
  if (!SimulateIsActive())
    return;
  SimulateThreadRegisterImpl(thread_handle);
}

ALWAYS_INLINE void SimulateBeforeChildThreadRuns() {
  if (!SimulateIsActive())
    return;
  SimulateBeforeChildThreadRunsImpl();
}

ALWAYS_INLINE void SimulateThreadFinish() {
  if (!SimulateIsActive())
    return;
  SimulateThreadFinishImpl();
}

// Mutex/cv ops

void SimulateMutexBlockImpl(uptr mutex_addr);
void SimulateMutexUnblockImpl(uptr mutex_addr);
void SimulateCondSignalImpl(uptr cond_addr);
void SimulateCondBroadcastImpl(uptr cond_addr);

ALWAYS_INLINE void SimulateMutexBlock(uptr mutex_addr) {
  if (!SimulateIsActive())
    return;
  SimulateMutexBlockImpl(mutex_addr);
}

ALWAYS_INLINE void SimulateMutexUnblock(uptr mutex_addr) {
  if (!SimulateIsActive())
    return;
  SimulateMutexUnblockImpl(mutex_addr);
}

ALWAYS_INLINE void SimulateCondSignal(uptr cond_addr) {
  if (!SimulateIsActive())
    return;
  SimulateCondSignalImpl(cond_addr);
}

ALWAYS_INLINE void SimulateCondBroadcast(uptr cond_addr) {
  if (!SimulateIsActive())
    return;
  SimulateCondBroadcastImpl(cond_addr);
}

bool SimulateJoinBlockImpl(uptr thread_handle);
void SimulateJoinResumeImpl();
template <class JoinFunction>
int SimulateJoin(void* th, void** ret, JoinFunction join_function) {
  bool sim_blocked = SimulateJoinBlockImpl((uptr)th);
  int res = join_function(th, ret);
  if (sim_blocked)
    SimulateJoinResumeImpl();
  return res;
}

struct ThreadState;
int SimulateCondWait(ThreadState* thr, uptr pc, void* c, void* m);

ALWAYS_INLINE void SimulateReportUnsupported(const char* func_name) {
  if (LIKELY(!SimulateIsActive()))
    return;
  SimulateReportUnsupportedImpl(func_name);
}

ALWAYS_INLINE void SimulateReportRace() {
  if (!SimulateIsActive())
    return;
  SimulateReportRaceImpl();
}

}  // namespace __tsan

#endif  // TSAN_SIMULATE_H
