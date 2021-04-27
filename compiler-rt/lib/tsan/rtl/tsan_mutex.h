//===-- tsan_mutex.h --------------------------------------------*- C++ -*-===//
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
#ifndef TSAN_MUTEX_H
#define TSAN_MUTEX_H

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_mutex.h"
#include "sanitizer_common/sanitizer_thread_safety.h"
#include "tsan_defs.h"

namespace __tsan {

enum {
  MutexTypeReport = MutexLastCommon,
  MutexTypeSyncVar,
  MutexTypeAnnotations,
  MutexTypeFired,
  MutexTypeRacy,
  MutexTypeGlobalProc,
  MutexTypeTrace,
  MutexTypeSlot,
  MutexTypeSlots,
};

/*
class MUTEX Mutex {
public:
  explicit Mutex(MutexType type);
  ~Mutex();

  void Lock() ACQUIRE();
  void Unlock() RELEASE();

  void ReadLock() ACQUIRE_SHARED();
  void ReadUnlock() RELEASE_SHARED();

  void CheckLocked() CHECK_LOCKED;

private:
  atomic_uintptr_t state_;
#if SANITIZER_DEBUG
  MutexType type_;
#endif

  Mutex(const Mutex&);
  void operator = (const Mutex&);
};

const uptr kUnlocked = 0;
const uptr kWriteLock = 1;
const uptr kReadLock = 2;

class Backoff {
public:
  Backoff() : iter_() {
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

#if SANITIZER_DEBUG && !SANITIZER_GO
void DebugMutexLock(MutexType type);
void DebugMutexUnlock(MutexType type);
#endif

ALWAYS_INLINE
Mutex::Mutex(MutexType type) {
  CHECK_GT(type, MutexTypeInvalid);
  CHECK_LT(type, MutexTypeCount);
#if SANITIZER_DEBUG
  type_ = type;
#endif
  atomic_store(&state_, kUnlocked, memory_order_relaxed);
}

ALWAYS_INLINE
Mutex::~Mutex() {
  CHECK_EQ(atomic_load(&state_, memory_order_relaxed), kUnlocked);
}

ALWAYS_INLINE
void Mutex::Lock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  DebugMutexLock(type_);
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

ALWAYS_INLINE
void Mutex::Unlock() {
  uptr prev = atomic_fetch_sub(&state_, kWriteLock, memory_order_release);
  (void)prev;
  DCHECK_NE(prev & kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  DebugMutexUnlock(type_);
#endif
}

ALWAYS_INLINE
void Mutex::ReadLock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  DebugMutexLock(type_);
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

ALWAYS_INLINE
void Mutex::ReadUnlock() {
  uptr prev = atomic_fetch_sub(&state_, kReadLock, memory_order_release);
  (void)prev;
  DCHECK_EQ(prev & kWriteLock, 0);
  DCHECK_GT(prev & ~kWriteLock, 0);
#if SANITIZER_DEBUG && !SANITIZER_GO
  DebugMutexUnlock(type_);
#endif
}

ALWAYS_INLINE
void Mutex::CheckLocked() {
  CHECK_NE(atomic_load(&state_, memory_order_relaxed), 0);
}
*/

typedef GenericScopedLock<Mutex> Lock;
typedef GenericScopedReadLock<Mutex> ReadLock;
typedef GenericScopedRWLock<Mutex> RWLock;

class InternalDeadlockDetector {
 public:
  InternalDeadlockDetector();
  void Lock(MutexType t, uptr pc);
  void Unlock(MutexType t);
  void CheckNoLocks();
 private:
};

void InitializeMutex();
void DebugCheckNoLocks();

// Checks if the current thread hold any runtime locks
// (e.g. when returning from an interceptor).
ALWAYS_INLINE void CheckNoLocks() {
#if SANITIZER_DEBUG && !SANITIZER_GO
  DebugCheckNoLocks();
#endif
}

}  // namespace __tsan

#endif  // TSAN_MUTEX_H
