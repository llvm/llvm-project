//===-- tsan_simulate.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
//
// TODO:
//   - Support calling __tsan_simulate while other threads are running.
//   - Support custom mutexes (e.g. absl::Mutex) and direct futex calls in
//     simulation mode. Currently, only pthread mutexes and condition variables
//     are supported; other synchronization primitives abort the simulation.
//   - Support timed operations (pthread_cond_timedwait,
//     pthread_mutex_timedlock, pthread_timedjoin_np) by treating them as their
//     non-timed counterparts.
//   - Support spinlocks (pthread_spin_lock/trylock/unlock) using the same
//     trylock-loop pattern as pthread_mutex_lock.
//   - Support rwlocks (pthread_rwlock_*) with reader/writer tracking in the
//     scheduler.
//   - Support barriers (pthread_barrier_wait) with a rendezvous mechanism in
//     the scheduler.
//   - Embed Waitset into the Sync object to replace the linear WaitsetMap.

#include "tsan_simulate.h"

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "tsan_flags.h"
#include "tsan_rtl.h"

extern "C" void* pthread_self();
DECLARE_REAL(int, pthread_mutex_unlock, void* m)
DECLARE_REAL(int, pthread_mutex_trylock, void* m)
namespace __tsan {

static constexpr int kMaxSimThreads = 64;

static int sim_current_iteration = 0;

static void PrintReproduceHint(int iteration) {
  Printf(
      "ThreadSanitizer: to reproduce, set "
      "TSAN_OPTIONS=simulate_scheduler=%s:simulate_start_iteration=%d:"
      "simulate_schedule_probability=%d\n",
      flags()->simulate_scheduler, iteration,
      flags()->simulate_schedule_probability);
}

static atomic_uint32_t sim_race_detected;

void SimulateReportUnsupportedImpl(const char* func_name) {
  Printf(
      "ThreadSanitizer: simulation error - unsupported interceptor called: "
      "%s at iteration %d\n"
      "Simulation does not support this synchronization primitive.\n",
      func_name, sim_current_iteration);
  PrintReproduceHint(sim_current_iteration);
  Die();
}

void SimulateReportRaceImpl() {
  atomic_store_relaxed(&sim_race_detected, 1);
  Printf("ThreadSanitizer: data race detected at iteration %d\n",
         sim_current_iteration);
}

void SimulateReportDeadlock() {
  Printf(
      "ThreadSanitizer: deadlock detected at iteration %d - all threads are "
      "blocked\n",
      sim_current_iteration);
  PrintReproduceHint(sim_current_iteration);
  Die();
}

static void SimulateReportMaxDepth(int max_depth, bool after_callback) {
  Printf("ThreadSanitizer: simulation hit max depth %d at iteration %d\n",
         max_depth, sim_current_iteration);
  if (after_callback)
    Printf(
        "Threads created by the callback were still running after the "
        "callback returned. They must finish without further action from the "
        "callback thread.\n");
  PrintReproduceHint(sim_current_iteration);
  Die();
}

namespace {

struct SimThread {
  enum State : u32 {
    Unused = 0,
    Runnable,  // Runnable - may be selected by the scheduler.
    Blocked,  // Blocked on mutex/condvar - scheduler must not pick this thread.
    Finished,  // Thread has exited the simulation.
    // Callback thread after the callback returned, parked until every other
    // thread has finished. Not counted as blocked for deadlock detection.
    WaitingForThreads,
  };

  Semaphore sem;
  State state;
  uptr thread_handle;  // This thread's pthread_t (from pthread_self())
  uptr joining_on;     // pthread_t this thread is joining on (0 if not joining)
};

// Waitset: tracks threads blocked waiting for a resource (mutex or condvar).
struct Waitset {
  static constexpr int kMaxWaiters = kMaxSimThreads;
  int waiters[kMaxWaiters];
  int count;

  Waitset() { Reset(); }

  void Reset() {
    count = 0;
    internal_memset(waiters, 0, sizeof(waiters));
  }

  void AddWaiter(int thread_idx) {
    CHECK_LT(count, kMaxWaiters);
    waiters[count++] = thread_idx;
  }

  // Randomly select and remove one thread from the waitset.
  int RemoveOne(u32* rng_state) {
    CHECK_GT(count, 0);
    int idx = RandN(rng_state, count);
    int thread_idx = waiters[idx];
    waiters[idx] = waiters[--count];
    return thread_idx;
  }

  // Remove all threads and return count.
  int RemoveAll(int* out_threads) {
    int n = count;
    for (int i = 0; i < count; i++) out_threads[i] = waiters[i];
    count = 0;
    return n;
  }
};

struct WaitsetMap {
  struct Element {
    uptr addr;
    Waitset waitset;
  };

  static constexpr int kMaxElements = 256;
  Element elements[kMaxElements];
  int count = 0;

  Waitset* Find(uptr addr) {
    for (int i = 0; i < count; i++)
      if (elements[i].addr == addr)
        return &elements[i].waitset;
    return nullptr;
  }

  Waitset* GetOrCreate(uptr addr) {
    Waitset* ws = Find(addr);
    if (ws)
      return ws;

    CHECK_LT(count, kMaxElements);
    int idx = count++;
    elements[idx].addr = addr;
    elements[idx].waitset.Reset();
    return &elements[idx].waitset;
  }

  void Reset() { count = 0; }
};

// SimScheduler controls which thread runs at each scheduling point. Exactly one
// thread is designated as "current" and executes user code. Other runnable
// threads park on their per-thread semaphore until the scheduler selects them.
class SimScheduler {
 public:
  SimScheduler() : current_(-1), thread_count_(0), depth_(0) {
    internal_memset(threads_, 0, sizeof(threads_));
  }

  void ResetForIteration() {
    current_ = -1;
    thread_count_ = 0;
    depth_ = 0;
    callback_waiter_ = -1;
    internal_memset(threads_, 0, sizeof(threads_));
    mutex_waitsets_.Reset();
    cond_waitsets_.Reset();
  }

  void StartIteration(u32 seed) {
    rng_state_ = seed;
    depth_ = 0;
    current_ = 0;
  }

  // ------- Main scheduling point -------
  //
  // Called by the currently running thread. May randomly switch to another
  // runnable thread.
  void Schedule(int caller_idx) {
    CHECK_EQ(caller_idx, current_);

    int max_depth = flags()->simulate_max_depth;
    if (++depth_ > max_depth)
      SimulateReportMaxDepth(max_depth, callback_waiter_ >= 0);

    int runnable = CountRunnable();
    if (runnable <= 1)
      return;

    int chosen = PickRandomRunnable(runnable);

    DumpStates(chosen, caller_idx);

    if (chosen == caller_idx)
      return;

    current_ = chosen;
    threads_[chosen].sem.Post();
    threads_[caller_idx].sem.Wait();
  }

  // Thread lifecycle methods

  int RegisterThread() {
    if (thread_count_ >= kMaxSimThreads) {
      Printf(
          "ThreadSanitizer: simulation error - max thread count %d exceeded\n",
          kMaxSimThreads);
      Die();
    }

    int idx = thread_count_++;
    threads_[idx].state = SimThread::Runnable;
    threads_[idx].thread_handle = 0;
    threads_[idx].joining_on = 0;
    return idx;
  }

  void SetThreadHandle(int idx, uptr handle) {
    threads_[idx].thread_handle = handle;
  }

  void ThreadStart(int idx) {
    CHECK_NE(current_, -1);
    threads_[idx].sem.Wait();
  }

  void ThreadFinish(int idx) {
    threads_[idx].state = SimThread::Finished;
    uptr my_handle = threads_[idx].thread_handle;
    CHECK_NE(my_handle, 0);

    if (my_handle != 0) {
      for (int i = 0; i < thread_count_; i++) {
        if (threads_[i].joining_on == my_handle) {
          threads_[i].state = SimThread::Runnable;
          threads_[i].joining_on = 0;
        }
      }
    }

    // Clear the handle to allow pthread_t reuse
    threads_[idx].thread_handle = 0;

    if (idx != current_)
      return;

    // We were current. Pick next runnable thread.
    PickNextAndWake();
  }

  // Called by the callback thread after the callback returns. Runs the other
  // threads until all of them have finished, then marks the callback thread
  // finished. Reports a deadlock if the remaining threads are all blocked.
  void WaitForThreadsAfterCallback(int caller_idx) {
    CHECK_EQ(caller_idx, current_);
    threads_[caller_idx].state = SimThread::WaitingForThreads;
    callback_waiter_ = caller_idx;

    if (CountRunnable() > 0) {
      PickNextAndWake();
      threads_[caller_idx].sem.Wait();
    } else if (CountBlocked() > 0) {
      SimulateReportDeadlock();
    }

    callback_waiter_ = -1;
    threads_[caller_idx].state = SimThread::Finished;
  }

  // ------- Blocking-call support -------

  // Called BEFORE a pthread_join call. Records the target pthread_t handle.
  void BeforeJoinCall(int idx, uptr target_handle) {
    threads_[idx].state = SimThread::Blocked;
    threads_[idx].joining_on = target_handle;

    if (idx == current_) {
      PickNextAndWake();
    }
  }

  // Check if a thread with the given pthread_t handle is still active
  // (i.e., not Finished). Returns false if thread not found or already
  // finished.
  bool IsThreadActive(uptr thread_handle) {
    for (int i = 0; i < thread_count_; i++) {
      if (threads_[i].state == SimThread::Finished)
        continue;
      if (threads_[i].thread_handle == thread_handle)
        return true;
    }
    return false;
  }

  int GetThreadCount() const { return thread_count_; }

  void MutexBlock(int caller_idx, uptr mutex_addr) {
    CHECK_EQ(caller_idx, current_);

    Waitset* ws = mutex_waitsets_.GetOrCreate(mutex_addr);
    ws->AddWaiter(caller_idx);

    threads_[caller_idx].state = SimThread::Blocked;

    PickNextAndWake();

    // Park this thread until woken by unlock.
    threads_[caller_idx].sem.Wait();
  }

  void MutexUnblock(uptr mutex_addr) {
    Waitset* ws = mutex_waitsets_.Find(mutex_addr);

    if (!ws || ws->count == 0)
      return;

    // Remove one waiter randomly and mark it as runnable.
    int thread_idx = ws->RemoveOne(&rng_state_);
    threads_[thread_idx].state = SimThread::Runnable;

    // If no thread is current, make the unblocked thread current and wake it.
    if (current_ == -1) {
      current_ = thread_idx;
      threads_[thread_idx].sem.Post();
    }
    // Otherwise it will be picked up by next Schedule() or when current
    // finishes.
  }

  void CondWait(int caller_idx, uptr cond_addr, uptr mutex_addr) {
    CHECK_EQ(caller_idx, current_);
    if (caller_idx != current_)
      return;

    // Add this thread to the condvar's waitset.
    Waitset* ws = cond_waitsets_.GetOrCreate(cond_addr);
    ws->AddWaiter(caller_idx);

    // Mark thread as blocked.
    threads_[caller_idx].state = SimThread::Blocked;

    // Pick next runnable thread and wake it.
    PickNextAndWake();

    // Park this thread until woken by signal/broadcast.
    threads_[caller_idx].sem.Wait();
  }

  void CondSignal(uptr cond_addr) {
    CHECK_NE(current_, -1);

    Waitset* ws = cond_waitsets_.Find(cond_addr);

    if (!ws || ws->count == 0)
      return;

    // Remove one waiter randomly and mark it as runnable.
    int thread_idx = ws->RemoveOne(&rng_state_);
    threads_[thread_idx].state = SimThread::Runnable;
  }

  void CondBroadcast(uptr cond_addr) {
    CHECK_NE(current_, -1);

    Waitset* ws = cond_waitsets_.Find(cond_addr);

    if (!ws || ws->count == 0)
      return;

    int woken[kMaxSimThreads];
    int n = ws->RemoveAll(woken);
    for (int i = 0; i < n; i++) threads_[woken[i]].state = SimThread::Runnable;
  }

  bool ShouldSchedule() {
    int schedule_probability = flags()->simulate_schedule_probability;
    if (schedule_probability >= 100)
      return true;
    if (schedule_probability <= 0)
      return false;
    u32 rand_val = RandN(&rng_state_, 100);
    return rand_val < static_cast<u32>(schedule_probability);
  }

 private:
  int CountInState(SimThread::State state) const {
    int n = 0;
    for (int i = 0; i < thread_count_; i++) {
      if (threads_[i].state == state)
        n++;
    }
    return n;
  }

  int CountRunnable() const { return CountInState(SimThread::Runnable); }

  int CountBlocked() const { return CountInState(SimThread::Blocked); }

  int PickRandomRunnable(int runnable) {
    int target = RandN(&rng_state_, runnable);
    for (int i = 0; i < thread_count_; i++) {
      if (threads_[i].state == SimThread::Runnable) {
        if (target == 0)
          return i;
        target--;
      }
    }
    CHECK(false);  // should not reach here
    return -1;
  }

  // Picks the next runnable thread and posts its semaphore. If none are
  // runnable and none are blocked, wakes the callback thread waiting in
  // WaitForThreadsAfterCallback, or sets current_ = -1 if there is none.
  void PickNextAndWake() {
    int runnable = CountRunnable();
    if (runnable == 0) {
      current_ = -1;
      if (CountBlocked() > 0)
        SimulateReportDeadlock();

      if (callback_waiter_ >= 0) {
        current_ = callback_waiter_;
        threads_[callback_waiter_].sem.Post();
      }
      return;
    }

    int chosen = PickRandomRunnable(runnable);
    current_ = chosen;
    threads_[chosen].sem.Post();
  }

  void DumpStates(int chosen = -1, int current = -1) {
    if (common_flags()->verbosity >= 2) {
      if (chosen >= 0) {
        Printf("Chose tid %d to run", chosen);
        if (current >= 0)
          Printf(" (current %d)", current);
        Printf(" - ");
      }
      Printf("Thread states: ");
      for (int i = 0; i < thread_count_; i++) {
        const char* state_str = "?";
        switch (threads_[i].state) {
          case SimThread::Unused:
            state_str = "Unused";
            break;
          case SimThread::Runnable:
            state_str = "Runnable";
            break;
          case SimThread::Blocked:
            state_str = "Blocked";
            break;
          case SimThread::Finished:
            state_str = "Finished";
            break;
          case SimThread::WaitingForThreads:
            state_str = "WaitingForThreads";
            break;
        }
        Printf("[%d:%s] ", i, state_str);
      }
      Printf("\n");
    }
  }

 private:
 public:
  u32 rng_state_ = 0;
  SimThread threads_[kMaxSimThreads];
  int current_;
  int thread_count_;
  int depth_;
  // Index of the thread in WaitForThreadsAfterCallback, or -1.
  int callback_waiter_ = -1;

  // Resource waitsets: map from resource address to waitset.
  WaitsetMap mutex_waitsets_;
  WaitsetMap cond_waitsets_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------

bool sim_active;

// Pointer to the current scheduler instance (valid while sim_active == true).
static SimScheduler* sim_sched;

class SimStateGuard {
  SimScheduler* sched_;

 public:
  SimStateGuard(SimScheduler* sched) : sched_(sched) { sim_active = true; }
  ~SimStateGuard() {
    sim_active = false;
    cur_thread()->sim_thread_idx = -1;
    sim_sched = nullptr;
    if (sched_) {
      sched_->~SimScheduler();
      InternalFree(sched_);
    }
  }
  SimStateGuard(const SimStateGuard&) = delete;
  SimStateGuard& operator=(const SimStateGuard&) = delete;
};

void SimulateScheduleImpl() {
  ThreadState* thr = cur_thread();
  CHECK_GE(thr->sim_thread_idx, 0);
  if (!sim_sched->ShouldSchedule())
    return;

  if (flags()->simulate_print_schedule_stacks) {
    Printf("=========== Schedule point (thread %d) ===========\n",
           thr->sim_thread_idx);
    PrintCurrentStack(thr, StackTrace::GetCurrentPc());
    Printf("==================================================\n");
  }

  CHECK_GE(thr->sim_thread_idx, 0);
  sim_sched->Schedule(thr->sim_thread_idx);
}

void SimulateThreadRegisterImpl(uptr thread_handle) {
  ThreadState* thr = cur_thread();
  thr->sim_thread_idx = sim_sched->RegisterThread();
  sim_sched->SetThreadHandle(thr->sim_thread_idx, thread_handle);
}

void SimulateBeforeChildThreadRunsImpl() {
  ThreadState* thr = cur_thread();
  CHECK_GE(thr->sim_thread_idx, 0);
  sim_sched->ThreadStart(thr->sim_thread_idx);
}

void SimulateThreadFinishImpl() {
  ThreadState* thr = cur_thread();
  int idx = thr->sim_thread_idx;
  CHECK_GE(idx, 0);
  thr->sim_thread_idx = -1;
  sim_sched->ThreadFinish(idx);
}

bool SimulateJoinBlockImpl(uptr thread_handle) {
  ThreadState* thr = cur_thread();
  CHECK_GE(thr->sim_thread_idx, 0);
  // Only mark ourselves as blocked if the target thread is still active.
  // If it's already finished, pthread_join will return immediately.
  if (sim_sched->IsThreadActive(thread_handle)) {
    sim_sched->BeforeJoinCall(thr->sim_thread_idx, thread_handle);
    return true;
  }
  return false;
}

void SimulateJoinResumeImpl() {
  // After BLOCK_REAL(pthread_join) returns, the target thread's ThreadFinish
  // marked us as Runnable and PickNextAndWake may have posted our semaphore.
  // We must consume that post to re-sync with the scheduler, otherwise the
  // pending post causes a future sem.Wait() to return spuriously, allowing
  // two threads to run simultaneously.
  sim_sched->threads_[cur_thread()->sim_thread_idx].sem.Wait();
}

void SimulateMutexBlockImpl(uptr mutex_addr) {
  ThreadState* thr = cur_thread();
  CHECK_GE(thr->sim_thread_idx, 0);
  sim_sched->MutexBlock(thr->sim_thread_idx, mutex_addr);
}

void SimulateMutexUnblockImpl(uptr mutex_addr) {
  sim_sched->MutexUnblock(mutex_addr);
}

void SimulateCondSignalImpl(uptr cond_addr) {
  sim_sched->CondSignal(cond_addr);
}

void SimulateCondBroadcastImpl(uptr cond_addr) {
  sim_sched->CondBroadcast(cond_addr);
}

int CheckForErrors(int iter, int start_iter) {
  if (atomic_load_relaxed(&sim_race_detected)) {
    PrintReproduceHint(iter);
    Printf(
        "ThreadSanitizer: simulation stopped due to race detection after %d "
        "iterations\n",
        iter - start_iter + 1);
    return -1;
  }

  // Check TSan's thread registry for threads that finished but were never
  // joined or detached (thread leaks).
  {
    bool has_leak = false;
    ThreadRegistryLock l(&ctx->thread_registry);
    ctx->thread_registry.RunCallbackForEachThreadLocked(
        [](ThreadContextBase* tctx_base, void* arg) {
          auto* tctx = static_cast<ThreadContext*>(tctx_base);
          if (tctx->detached || tctx->status != ThreadStatusFinished)
            return;
          *static_cast<bool*>(arg) = true;
        },
        &has_leak);
    if (has_leak) {
      Printf("ThreadSanitizer: thread leak detected at iteration %d\n", iter);
      PrintReproduceHint(iter);
      Printf(
          "ThreadSanitizer: simulation stopped due to thread leak after %d "
          "iterations\n",
          iter - start_iter + 1);
      return -1;
    }
  }

  return 0;
}

int SimulateRun(void (*callback)(void*), void* arg) {
  if (SimulateIsActive()) {
    Printf(
        "ThreadSanitizer: __tsan_simulate cannot be called while a "
        "simulation is running. Not running callback\n");
    return -1;
  }

  const char* sched = flags()->simulate_scheduler;
  if (!sched || !sched[0]) {
    callback(arg);
    return 0;
  }
  if (internal_strcmp(sched, "random") != 0) {
    Printf(
        "ThreadSanitizer: unknown simulate_scheduler '%s' (supported values: "
        "random). Not running callback\n",
        sched);
    return -1;
  }

  uptr running_threads = 0;
  ctx->thread_registry.GetNumberOfThreads(nullptr, &running_threads, nullptr);
  if (running_threads > 1) {
    Printf(
        "ThreadSanitizer: simulation cannot start - other threads are "
        "running (%zu threads detected).\n"
        "Simulation requires that only the calling thread exists. "
        "Not running callback\n",
        running_threads);
    return -1;
  }

  atomic_store_relaxed(&sim_race_detected, 0);

  int iterations = flags()->simulate_iterations;
  if (iterations <= 0) {
    Printf("ThreadSanitizer: simulate_iterations must be > 0 (got %d)\n",
           iterations);
    return -1;
  }

  int start_iter = flags()->simulate_start_iteration;
  if (start_iter < 0) {
    Printf("ThreadSanitizer: simulate_start_iteration must be >= 0 (got %d)\n",
           start_iter);
    return -1;
  }

  int prob = flags()->simulate_schedule_probability;
  if (prob < 0 || prob > 100) {
    Printf(
        "ThreadSanitizer: simulate_schedule_probability must be >=0 and <= 100 "
        "(got %d)\n",
        prob);
    return -1;
  }

  int max_depth = flags()->simulate_max_depth;
  Printf(
      "ThreadSanitizer: simulation starting (iterations %d..%d, max_depth=%d, "
      "scheduler=%s)\n",
      start_iter, start_iter + iterations - 1, max_depth, sched);

  void* sched_mem = InternalAlloc(sizeof(SimScheduler));
  SimScheduler* sched_ptr = new (sched_mem) SimScheduler();
  sim_sched = sched_ptr;

  SimStateGuard guard(sched_ptr);

  for (int iter = start_iter; iter < start_iter + iterations; iter++) {
    sim_current_iteration = iter;

    sched_ptr->ResetForIteration();

    int main_idx = sched_ptr->RegisterThread();
    CHECK_EQ(main_idx, 0);
    cur_thread()->sim_thread_idx = main_idx;
    sched_ptr->SetThreadHandle(main_idx, (uptr)pthread_self());

    sched_ptr->StartIteration(iter);

    DPrintf(1, "Start callback iter=%d\n", iter);
    callback(arg);
    DPrintf(1, "End callback iter=%d\n", iter);
    sched_ptr->WaitForThreadsAfterCallback(main_idx);

    if (iter == start_iter && sched_ptr->GetThreadCount() == 1) {
      Printf("ThreadSanitizer: simulation exiting - no threads were spawned\n");
      return 0;
    }

    if (int rc = CheckForErrors(iter, start_iter); rc)
      return rc;
  }

  Printf("ThreadSanitizer: simulation finished (%d iterations)\n", iterations);
  return 0;
}

int SimulateCondWait(ThreadState* thr, uptr pc, void* c, void* m) {
  int res = REAL(pthread_mutex_unlock)(m);
  CHECK_EQ(res, 0);

  SimulateMutexUnblockImpl((uptr)m);

  int idx = cur_thread()->sim_thread_idx;
  CHECK_GE(idx, 0);
  sim_sched->CondWait(idx, (uptr)c, (uptr)m);

  // After waking, re-acquire the mutex (mimicking pthread_cond_wait
  // behavior).
  SimulateSchedule();
  while (true) {
    res = REAL(pthread_mutex_trylock)(m);
    if (res == 0 || res == errno_EOWNERDEAD)
      break;
    if (res != errno_EBUSY) {
      // Some other error - give up.
      MutexPostLock(thr, pc, (uptr)m, MutexFlagDoPreLockOnPostLock);
      return res;
    }
    SimulateMutexBlockImpl((uptr)m);
  }
  return res;
}

}  // namespace __tsan
