//===-- sanitizer_mutex.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer/AddressSanitizer runtime.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_MUTEX_H
#define SANITIZER_MUTEX_H

#include "sanitizer_atomic.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_libc.h"
#include "sanitizer_thread_safety.h"

namespace __sanitizer {

class StaticSpinMutex {
 public:
  StaticSpinMutex() = default;
 
  void Init() {
    atomic_store(&state_, 0, memory_order_relaxed);
  }

  void Lock() {
    if (UNLIKELY(!TryLock()))
      LockSlow();
  }

  bool TryLock() {
    return atomic_exchange(&state_, 1, memory_order_acquire) == 0;
  }

  void Unlock() {
    atomic_store(&state_, 0, memory_order_release);
  }

  void CheckLocked() const {
    CHECK_EQ(atomic_load(&state_, memory_order_relaxed), 1);
  }

 private:
  atomic_uint8_t state_;

  void LockSlow();

  StaticSpinMutex(const StaticSpinMutex&) = delete;
  void operator = (const StaticSpinMutex&) = delete;
};

class SpinMutex : public StaticSpinMutex {
 public:
  SpinMutex() {
    Init();
  }
};

typedef int MutexType;

enum {
  MutexInvalid = 0,
  MutexThreadRegistry,
  MutexLastCommon,
  // Type for legacy mutexes that are not checked for deadlocks.
  MutexUnchecked = -1,
  // The leaf mutexes can be locked under any other mutex,
  // but no other mutex can be locked while under a leaf mutex.
  MutexLeaf = -1,
  // Multiple mutexes of this type can be locked at the same time.
  MutexMulti = -3,
};

#if SANITIZER_DEBUG && !SANITIZER_GO
void DebugMutexLock(MutexType type, uptr pc);
void DebugMutexUnlock(MutexType type);

struct MutexMeta {
  MutexType type;
  const char* name;
  // The table fixes what mutexes can be locked under what mutexes.
  // E.g. if the entry for MutexTypeFoo contains MutexTypeBar,
  // then Bar mutex can be locked while under Foo mutex.
  // The MutexTypeLeaf mutexes can be locked under any other mutexes,
  // but can't lock any other mutexes.
  MutexType can_lock[10];
};
#endif

class CheckedMutex {
protected:
  CheckedMutex(MutexType type) {
#if SANITIZER_DEBUG && !SANITIZER_GO
    type_ = type;
#endif
  }

  void DebugLock(uptr pc) {
#if SANITIZER_DEBUG && !SANITIZER_GO
    DebugMutexLock(type_, pc);
#endif
  }

  void DebugUnlock() {
#if SANITIZER_DEBUG && !SANITIZER_GO
    DebugMutexUnlock(type_);
#endif
  }

private:
#if SANITIZER_DEBUG && !SANITIZER_GO
  MutexType type_;
#endif
};

// Semaphore provides destruction-safety:
// last thread returned from Wait can destroy the object.
class Semaphore {
 public:
  Semaphore();
  Semaphore(const Semaphore &) = delete;
  void operator=(const Semaphore &) = delete;

  void Wait();
  void Post(u32 count = 1);

 private:
#if SANITIZER_LINUX
  atomic_uint32_t state_;
#else
  uptr state_;
#endif
};

// Reader-writer mutex.
class MUTEX Mutex : public CheckedMutex {
 public:
  Mutex(MutexType type = MutexUnchecked)
    : CheckedMutex(type) {
    atomic_store_relaxed(&state_, 0);
  }

  ~Mutex() {
    DCHECK_EQ(atomic_load_relaxed(&state_), 0);
  }

  void Lock() ACQUIRE() {
    DebugLock(GET_CALLER_PC());
    u64 reset_mask = ~0ull;
    u64 state = atomic_load_relaxed(&state_);
    const uptr kMaxSpinIters = 1500;
    for (uptr spin_iters = 0;;spin_iters++) {
      u64 new_state;
      bool locked = (state & (kWriterLock | kReaderLockMask)) != 0;
      if (LIKELY(!locked)) {
        new_state = (state | kWriterLock) & reset_mask;
      } else if (spin_iters > kMaxSpinIters) {
        new_state = (state + kWaitingWriterInc) & reset_mask;
      } else if ((state & kWriterWoken) == 0) {
        new_state = state | kWriterWoken;
      } else {
        state = atomic_load(&state_, memory_order_relaxed);
        continue;
      }
      if (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state, memory_order_acquire)))
        continue;
      if (LIKELY(!locked))
        return;
      if (spin_iters > kMaxSpinIters) {
        writers_.Wait();
        state = atomic_load(&state_, memory_order_relaxed);
        DCHECK_NE(state & kWriterWoken, 0);
      }
      reset_mask = ~kWriterWoken;
    }
  }

  void Unlock() RELEASE() {
    DebugUnlock();
    u64 state = atomic_load_relaxed(&state_);
    for (;;) {
      DCHECK_NE(state & kWriterLock, 0);
      DCHECK_EQ(state & kReaderLockMask, 0);
      u64 new_state = state & ~kWriterLock;
      bool wake_writer = (state & kWriterWoken) == 0 &&
          (state & kWaitingWriterMask) != 0;
      if (wake_writer)
        new_state = (new_state - kWaitingWriterInc) | kWriterWoken;
      u64 wake_readers = (state & (kWriterWoken | kWaitingWriterMask)) != 0 ? 0 :
          ((state & kWaitingReaderMask) >> kWaitingReaderShift);
      if (wake_readers)
        new_state = (new_state & ~ kWaitingReaderMask) + (wake_readers << kReaderLockShift);
      if (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state, memory_order_release)))
        continue;
      if (UNLIKELY(wake_writer))
        writers_.Post();
      else if(UNLIKELY(wake_readers))
        readers_.Post(wake_readers);
      return;
    }
  }

  void ReadLock() ACQUIRE_SHARED() {
    DebugLock(GET_CALLER_PC());
    u64 state = atomic_load_relaxed(&state_);
    for (;;) {
      u64 new_state;
      bool locked = (state & kReaderLockMask) == 0 &&
          (state & (kWriterLock | kWriterWoken | kWaitingWriterMask)) != 0;
      if (LIKELY(!locked))
        new_state = state + kReaderLockInc;
      else
        new_state = state + kWaitingReaderInc;
      if (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state, memory_order_acquire)))
        continue;
      if (UNLIKELY(locked))
        readers_.Wait();
      DCHECK_EQ(atomic_load_relaxed(&state_) & kWriterLock, 0);
      DCHECK_NE(atomic_load_relaxed(&state_) & kReaderLockMask, 0);
      return;
    }
  }

  void ReadUnlock() RELEASE_SHARED() {
    DebugUnlock();
    u64 state = atomic_load_relaxed(&state_);
    for (;;) {
      DCHECK_NE(state & kReaderLockMask, 0);
      DCHECK_EQ(state & (kWaitingReaderMask | kWriterLock), 0);
      u64 new_state = state - kReaderLockInc;
      bool wake = (state & kWriterWoken) == 0 &&
          (state & kWaitingWriterMask) != 0;
      if (wake)
        new_state = (new_state - kWaitingWriterInc) | kWriterWoken;
      if (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state, memory_order_release)))
        continue;
      if (UNLIKELY(wake))
        writers_.Post();
      return;
    }
  }

  // This function does not guarantee an explicit check that the calling thread
  // is the thread which owns the mutex. This behavior, while more strictly
  // correct, causes problems in cases like StopTheWorld, where a parent thread
  // owns the mutex but a child checks that it is locked. Rather than
  // maintaining complex state to work around those situations, the check only
  // checks that the mutex is owned, and assumes callers to be generally
  // well-behaved.
  void CheckLocked() const CHECK_LOCKED {
    CHECK_NE(atomic_load(&state_, memory_order_relaxed), 0);
  }

 private:
  atomic_uint64_t state_;
  Semaphore writers_;
  Semaphore readers_;

  static constexpr u64 kCounterWidth = 20;
  static constexpr u64 kReaderLockShift = 0;
  static constexpr u64 kReaderLockInc = 1ull << kReaderLockShift;
  static constexpr u64 kReaderLockMask = ((1ull << kCounterWidth) - 1) << kReaderLockShift;
  static constexpr u64 kWaitingReaderShift = kCounterWidth;
  static constexpr u64 kWaitingReaderInc = 1ull << kWaitingReaderShift;
  static constexpr u64 kWaitingReaderMask = ((1ull << kCounterWidth) - 1) << kWaitingReaderShift;
  static constexpr u64 kWaitingWriterShift = 2 * kCounterWidth;
  static constexpr u64 kWaitingWriterInc = 1ull << kWaitingWriterShift;
  static constexpr u64 kWaitingWriterMask = ((1ull << kCounterWidth) - 1) << kWaitingWriterShift;
  static constexpr u64 kWriterLock = 1ull << (3 * kCounterWidth);
  static constexpr u64 kWriterWoken = 1ull << (3 * kCounterWidth + 1);

  Mutex(const Mutex&) = delete;
  void operator = (const Mutex&) = delete;
};

typedef Mutex RWMutex;
typedef Mutex BlockingMutex;

template <typename MutexType>
class SCOPED_LOCK GenericScopedLock {
 public:
  ALWAYS_INLINE explicit GenericScopedLock(MutexType *mu) ACQUIRE(mu)
      : mu_(mu) {
    mu_->Lock();
  }

  ALWAYS_INLINE ~GenericScopedLock() RELEASE() { mu_->Unlock(); }

 private:
  MutexType *mu_;

  GenericScopedLock(const GenericScopedLock &) = delete;
  void operator=(const GenericScopedLock &) = delete;
};

template <typename MutexType>
class SCOPED_LOCK GenericScopedReadLock {
 public:
  ALWAYS_INLINE explicit GenericScopedReadLock(MutexType *mu) ACQUIRE_SHARED(mu)
      : mu_(mu) {
    mu_->ReadLock();
  }

  ALWAYS_INLINE ~GenericScopedReadLock() RELEASE() { mu_->ReadUnlock(); }

 private:
  MutexType *mu_;

  GenericScopedReadLock(const GenericScopedReadLock &) = delete;
  void operator=(const GenericScopedReadLock &) = delete;
};

template <typename MutexType>
class SCOPED_LOCK GenericScopedRWLock {
 public:
  ALWAYS_INLINE explicit GenericScopedRWLock(MutexType *mu, bool write)
      ACQUIRE(mu)
      : mu_(mu), write_(write) {
    if (write_)
      mu_->Lock();
    else
      mu_->ReadLock();
  }

  ALWAYS_INLINE ~GenericScopedRWLock() RELEASE() {
    if (write_)
      mu_->Unlock();
    else
      mu_->ReadUnlock();
  }

 private:
  MutexType *mu_;
  bool write_;

  GenericScopedRWLock(const GenericScopedRWLock &) = delete;
  void operator=(const GenericScopedRWLock &) = delete;
};

typedef GenericScopedLock<StaticSpinMutex> SpinMutexLock;
typedef GenericScopedLock<BlockingMutex> BlockingMutexLock;
typedef GenericScopedLock<RWMutex> RWMutexLock;
typedef GenericScopedReadLock<RWMutex> RWMutexReadLock;

}  // namespace __sanitizer

#endif  // SANITIZER_MUTEX_H
