//===-- trec_mutex.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TraceRecorder (TRec), a race detector.
//
//===----------------------------------------------------------------------===//
#include "sanitizer_common/sanitizer_libc.h"
#include "trec_mutex.h"
#include "trec_platform.h"
#include "trec_rtl.h"

namespace __trec {

// Simple reader-writer spin-mutex. Optimized for not-so-contended case.
// Readers have preference, can possibly starvate writers.

// The table fixes what mutexes can be locked under what mutexes.
// E.g. if the row for MutexTypeThreads contains MutexTypeReport,
// then Report mutex can be locked while under Threads mutex.
// The leaf mutexes can be locked under any other mutexes.
// Recursive locking is not supported.
#if SANITIZER_DEBUG && !SANITIZER_GO
const MutexType MutexTypeLeaf = (MutexType)-1;
static MutexType CanLockTab[MutexTypeCount][MutexTypeCount] = {
  /*0  MutexTypeInvalid*/     {},
  /*1  MutexTypeTrace*/       {MutexTypeLeaf},
  /*2  MutexTypeThreads*/     {MutexTypeReport},
  /*3  MutexTypeReport*/      {MutexTypeSyncVar,
                               MutexTypeMBlock, MutexTypeJavaMBlock},
  /*4  MutexTypeSyncVar*/     {MutexTypeDDetector},
  /*5  MutexTypeSyncTab*/     {},  // unused
  /*6  MutexTypeSlab*/        {MutexTypeLeaf},
  /*7  MutexTypeAnnotations*/ {},
  /*8  MutexTypeAtExit*/      {MutexTypeSyncVar},
  /*9  MutexTypeMBlock*/      {MutexTypeSyncVar},
  /*10 MutexTypeJavaMBlock*/  {MutexTypeSyncVar},
  /*11 MutexTypeDDetector*/   {},
  /*12 MutexTypeFired*/       {MutexTypeLeaf},
  /*13 MutexTypeRacy*/        {MutexTypeLeaf},
  /*14 MutexTypeGlobalProc*/  {},
};

static bool CanLockAdj[MutexTypeCount][MutexTypeCount];
#endif


InternalDeadlockDetector::InternalDeadlockDetector() {
  // Rely on zero initialization because some mutexes can be locked before ctor.
}

#if SANITIZER_DEBUG && !SANITIZER_GO
void InternalDeadlockDetector::Lock(MutexType t) {
  // Printf("LOCK %d @%zu\n", t, seq_ + 1);
  CHECK_GT(t, MutexTypeInvalid);
  CHECK_LT(t, MutexTypeCount);
  u64 max_seq = 0;
  u64 max_idx = MutexTypeInvalid;
  for (int i = 0; i != MutexTypeCount; i++) {
    if (locked_[i] == 0)
      continue;
    CHECK_NE(locked_[i], max_seq);
    if (max_seq < locked_[i]) {
      max_seq = locked_[i];
      max_idx = i;
    }
  }
  locked_[t] = ++seq_;
  if (max_idx == MutexTypeInvalid)
    return;
  // Printf("  last %d @%zu\n", max_idx, max_seq);
  if (!CanLockAdj[max_idx][t]) {
    Printf("TraceRecorder: internal deadlock detected\n");
    Printf("TraceRecorder: can't lock %d while under %zu\n",
               t, (uptr)max_idx);
    CHECK(0);
  }
}

void InternalDeadlockDetector::Unlock(MutexType t) {
  // Printf("UNLO %d @%zu #%zu\n", t, seq_, locked_[t]);
  CHECK(locked_[t]);
  locked_[t] = 0;
}

#endif


const uptr kUnlocked = 0;
const uptr kWriteLock = 1;
const uptr kReadLock = 2;

class Backoff {
 public:
  Backoff()
    : iter_() {
  }

  bool Do() {
    if (iter_++ < kActiveSpinIters)
      proc_yield(kActiveSpinCnt);
    else
      internal_sched_yield();
    return true;
  }

  u64 Contention() const {
    u64 active = iter_ % kActiveSpinIters;
    u64 passive = iter_ - active;
    return active + 10 * passive;
  }

 private:
  int iter_;
  static const int kActiveSpinIters = 10;
  static const int kActiveSpinCnt = 20;
};

Mutex::Mutex(MutexType type) {
  CHECK_GT(type, MutexTypeInvalid);
  CHECK_LT(type, MutexTypeCount);
#if SANITIZER_DEBUG
  type_ = type;
#endif
  atomic_store(&state_, kUnlocked, memory_order_relaxed);
}

Mutex::~Mutex() {
  CHECK_EQ(atomic_load(&state_, memory_order_relaxed), kUnlocked);
}

void Mutex::Lock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Lock(type_);
#endif
  uptr cmp = kUnlocked;
  if (atomic_compare_exchange_strong(&state_, &cmp, kWriteLock,
                                     memory_order_acquire))
    return;
  for (Backoff backoff; backoff.Do();) {
    if (atomic_load(&state_, memory_order_relaxed) == kUnlocked) {
      cmp = kUnlocked;
      if (atomic_compare_exchange_weak(&state_, &cmp, kWriteLock,
                                       memory_order_acquire)) {
        return;
      }
    }
  }
}

void Mutex::Unlock() {
  uptr prev = atomic_fetch_sub(&state_, kWriteLock, memory_order_release);
  (void)prev;
  DCHECK_NE(prev & kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Unlock(type_);
#endif
}

void Mutex::ReadLock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Lock(type_);
#endif
  uptr prev = atomic_fetch_add(&state_, kReadLock, memory_order_acquire);
  if ((prev & kWriteLock) == 0)
    return;
  for (Backoff backoff; backoff.Do();) {
    prev = atomic_load(&state_, memory_order_acquire);
    if ((prev & kWriteLock) == 0) {
      return;
    }
  }
}

void Mutex::ReadUnlock() {
  uptr prev = atomic_fetch_sub(&state_, kReadLock, memory_order_release);
  (void)prev;
  DCHECK_EQ(prev & kWriteLock, 0);
  DCHECK_GT(prev & ~kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  cur_thread()->internal_deadlock_detector.Unlock(type_);
#endif
}

void Mutex::CheckLocked() {
  CHECK_NE(atomic_load(&state_, memory_order_relaxed), 0);
}

}  // namespace __trec
