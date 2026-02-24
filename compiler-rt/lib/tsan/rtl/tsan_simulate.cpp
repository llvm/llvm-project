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

static atomic_uint32_t sim_max_depth_hit;
static atomic_uint32_t sim_race_detected;
static atomic_uint32_t sim_unsupported_interceptor_called;

void SimulateReportUnsupportedImpl(const char* func_name) {
  atomic_store_relaxed(&sim_unsupported_interceptor_called, 1);
  Printf(
      "ThreadSanitizer: simulation error - unsupported interceptor called: "
      "%s\n"
      "Simulation does not support this synchronization primitive.\n",
      func_name);
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
  Printf(
      "ThreadSanitizer: to reproduce, set "
      "TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=%d\n",
      sim_current_iteration);
  Die();
}

namespace {

struct SimThread {
  enum State : u32 {
    Unused = 0,
    Runnable,  // Runnable - may be selected by the scheduler.
    Blocked,  // Blocked on mutex/condvar - scheduler must not pick this thread.
    Finished,  // Thread has exited the simulation.
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
  // Matches Relacy's approach to maximize interleaving exploration.
  int RemoveOne(u32* rng_state) {
    CHECK_GT(count, 0);
    // Pick a random thread from the waitset.
    int idx = RandN(rng_state, count);
    int thread_idx = waiters[idx];
    // Remove it by shifting remaining threads.
    for (int i = idx + 1; i < count; i++) waiters[i - 1] = waiters[i];
    count--;
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
    if (atomic_load_relaxed(&sim_max_depth_hit))
      return;

    CHECK_EQ(caller_idx, current_);

    int max_depth = flags()->simulate_max_depth;
    if (++depth_ > max_depth) {
      atomic_store_relaxed(&sim_max_depth_hit, 1);
      Printf("ThreadSanitizer: simulation hit max depth %d at iteration %d\n",
             max_depth, sim_current_iteration);
    }

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

  // Called AFTER a blocking OS call returns. Marks this thread as Runnable
  // (runnable) again. If no thread is currently running, this thread becomes
  // current and returns immediately. Otherwise it parks until selected.
  void AfterBlockingCall(int idx) {
    threads_[idx].state = SimThread::Runnable;

    if (current_ == -1)
      current_ = idx;

    // Another thread is running. Park until selected.
    threads_[idx].sem.Wait();
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
    int schedule_probability_ = flags()->simulate_schedule_probability;
    if (schedule_probability_ >= 100)
      return true;
    if (schedule_probability_ <= 0)
      return false;
    u32 rand_val = RandN(&rng_state_, 100);
    return rand_val < static_cast<u32>(schedule_probability_);
  }

 private:
  int CountRunnable() const {
    int n = 0;
    for (int i = 0; i < thread_count_; i++) {
      if (threads_[i].state == SimThread::Runnable)
        n++;
    }
    return n;
  }

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

  // Picks the next runnable thread and posts its semaphore, or sets current_ =
  // -1 if none are runnable.
  void PickNextAndWake() {
    int runnable = CountRunnable();
    if (runnable == 0) {
      current_ = -1;
      int blocked = 0;
      for (int i = 0; i < thread_count_; i++)
        if (threads_[i].state == SimThread::Blocked)
          blocked++;

      if (blocked > 0)
        SimulateReportDeadlock();

      // Should only hapen when the callback thread is exiting
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

void SimulateThreadUnblockImpl() {
  ThreadState* thr = cur_thread();
  CHECK_GE(thr->sim_thread_idx, 0);
  sim_sched->AfterBlockingCall(thr->sim_thread_idx);
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

int CheckForErors(int iter, int start_iter) {
  if (atomic_load_relaxed(&sim_unsupported_interceptor_called)) {
    Printf("ThreadSanitizer: unsupported interceptor at iteration %d\n", iter);
    Printf(
        "ThreadSanitizer: to reproduce, set "
        "TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=%d\n",
        iter);
    Printf("ThreadSanitizer: simulation aborted after %d iterations\n",
           iter - start_iter + 1);
    return -1;
  }

  if (atomic_load_relaxed(&sim_max_depth_hit)) {
    Printf(
        "ThreadSanitizer: to reproduce, set "
        "TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=%d\n",
        iter);
    Printf(
        "ThreadSanitizer: simulation stopped due to max depth after %d "
        "iterations\n",
        iter - start_iter + 1);
    return -1;
  }

  if (atomic_load_relaxed(&sim_race_detected)) {
    Printf(
        "ThreadSanitizer: to reproduce, set "
        "TSAN_OPTIONS=simulate_scheduler=random:simulate_start_iteration=%d\n",
        iter);
    Printf(
        "ThreadSanitizer: simulation stopped due to race detection after %d "
        "iterations\n",
        iter - start_iter + 1);
    return -1;
  }

  return 0;
}

int SimulateRun(void (*callback)(void*), void* arg) {
  const char* sched = flags()->simulate_scheduler;
  if (!sched || !sched[0] || internal_strcmp(sched, "random") != 0) {
    callback(arg);
    return 0;
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

  atomic_store_relaxed(&sim_unsupported_interceptor_called, 0);
  atomic_store_relaxed(&sim_max_depth_hit, 0);
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
        "ThreadSanitizer: simulate_schedule_probabilitymust be >=0 and <= 100 "
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

    if (iter == start_iter && sched_ptr->GetThreadCount() == 1) {
      Printf("ThreadSanitizer: simulation exiting - no threads were spawned\n");
      return 0;
    }

    if (int rc = CheckForErors(iter, start_iter); rc)
      return rc;

    sched_ptr->ThreadFinish(main_idx);
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
