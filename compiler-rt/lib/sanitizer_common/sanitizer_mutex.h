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

class MUTEX StaticSpinMutex {
 public:
  void Init() {
    atomic_store(&state_, 0, memory_order_relaxed);
  }

  void Lock() ACQUIRE() {
    if (LIKELY(TryLock()))
      return;
    LockSlow();
  }

  bool TryLock() TRY_ACQUIRE(true) {
    return atomic_exchange(&state_, 1, memory_order_acquire) == 0;
  }

  void Unlock() RELEASE() { atomic_store(&state_, 0, memory_order_release); }

  void CheckLocked() const CHECK_LOCKED() {
    CHECK_EQ(atomic_load(&state_, memory_order_relaxed), 1);
  }

 private:
  atomic_uint8_t state_;

  void LockSlow();
};

class MUTEX SpinMutex : public StaticSpinMutex {
 public:
  SpinMutex() {
    Init();
  }

  SpinMutex(const SpinMutex &) = delete;
  void operator=(const SpinMutex &) = delete;
};

// Semaphore provides an OS-dependent way to park/unpark threads.
// The last thread returned from Wait can destroy the object
// (destruction-safety).
class Semaphore {
 public:
  constexpr Semaphore() {}
  Semaphore(const Semaphore &) = delete;
  void operator=(const Semaphore &) = delete;

  void Wait();
  void Post(u32 count = 1);

 private:
  atomic_uint32_t state_ = {0};
};

// Reader-writer mutex.
class MUTEX Mutex2 {
 public:
  constexpr Mutex2() {}

  void Lock() ACQUIRE() {
    u64 reset_mask = ~0ull;
    u64 state = atomic_load_relaxed(&state_);
    const uptr kMaxSpinIters = 1500;
    for (uptr spin_iters = 0;; spin_iters++) {
      u64 new_state;
      bool locked = (state & (kWriterLock | kReaderLockMask)) != 0;
      if (LIKELY(!locked)) {
        // The mutex is not read-/write-locked, try to lock.
        new_state = (state | kWriterLock) & reset_mask;
      } else if (spin_iters > kMaxSpinIters) {
        // We've spun enough, increment waiting writers count and block.
        // The counter will be decremented by whoever wakes us.
        new_state = (state + kWaitingWriterInc) & reset_mask;
      } else if ((state & kWriterSpinWait) == 0) {
        // Active spinning, but denote our presence so that unlocking
        // thread does not wake up other threads.
        new_state = state | kWriterSpinWait;
      } else {
        // Active spinning.
        state = atomic_load(&state_, memory_order_relaxed);
        continue;
      }
      if (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state,
                                                 memory_order_acquire)))
        continue;
      if (LIKELY(!locked))
        return;  // We've locked the mutex.
      if (spin_iters > kMaxSpinIters) {
        // We've incremented waiting writers, so now block.
        writers_.Wait();
        spin_iters = 0;
        state = atomic_load(&state_, memory_order_relaxed);
        DCHECK_NE(state & kWriterSpinWait, 0);
      } else {
        // We've set kWriterSpinWait, but we are still in active spinning.
      }
      // We either blocked and were unblocked,
      // or we just spun but set kWriterSpinWait.
      // Either way we need to reset kWriterSpinWait
      // next time we take the lock or block again.
      reset_mask = ~kWriterSpinWait;
    }
  }

  void Unlock() RELEASE() {
    bool wake_writer;
    u64 wake_readers;
    u64 new_state;
    u64 state = atomic_load_relaxed(&state_);
    do {
      DCHECK_NE(state & kWriterLock, 0);
      DCHECK_EQ(state & kReaderLockMask, 0);
      new_state = state & ~kWriterLock;
      wake_writer =
          (state & kWriterSpinWait) == 0 && (state & kWaitingWriterMask) != 0;
      if (wake_writer)
        new_state = (new_state - kWaitingWriterInc) | kWriterSpinWait;
      wake_readers =
          (state & (kWriterSpinWait | kWaitingWriterMask)) != 0
              ? 0
              : ((state & kWaitingReaderMask) >> kWaitingReaderShift);
      if (wake_readers)
        new_state = (new_state & ~kWaitingReaderMask) +
                    (wake_readers << kReaderLockShift);
    } while (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state,
                                                    memory_order_release)));
    if (UNLIKELY(wake_writer))
      writers_.Post();
    else if (UNLIKELY(wake_readers))
      readers_.Post(wake_readers);
  }

  void ReadLock() ACQUIRE_SHARED() {
    bool locked;
    u64 new_state;
    u64 state = atomic_load_relaxed(&state_);
    do {
      locked =
          (state & kReaderLockMask) == 0 &&
          (state & (kWriterLock | kWriterSpinWait | kWaitingWriterMask)) != 0;
      if (LIKELY(!locked))
        new_state = state + kReaderLockInc;
      else
        new_state = state + kWaitingReaderInc;
    } while (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state,
                                                    memory_order_acquire)));
    if (UNLIKELY(locked))
      readers_.Wait();
    DCHECK_EQ(atomic_load_relaxed(&state_) & kWriterLock, 0);
    DCHECK_NE(atomic_load_relaxed(&state_) & kReaderLockMask, 0);
  }

  void ReadUnlock() RELEASE_SHARED() {
    bool wake;
    u64 new_state;
    u64 state = atomic_load_relaxed(&state_);
    do {
      DCHECK_NE(state & kReaderLockMask, 0);
      DCHECK_EQ(state & (kWaitingReaderMask | kWriterLock), 0);
      new_state = state - kReaderLockInc;
      wake = (new_state & (kReaderLockMask | kWriterSpinWait)) == 0 &&
             (new_state & kWaitingWriterMask) != 0;
      if (wake)
        new_state = (new_state - kWaitingWriterInc) | kWriterSpinWait;
    } while (UNLIKELY(!atomic_compare_exchange_weak(&state_, &state, new_state,
                                                    memory_order_release)));
    if (UNLIKELY(wake))
      writers_.Post();
  }

  // This function does not guarantee an explicit check that the calling thread
  // is the thread which owns the mutex. This behavior, while more strictly
  // correct, causes problems in cases like StopTheWorld, where a parent thread
  // owns the mutex but a child checks that it is locked. Rather than
  // maintaining complex state to work around those situations, the check only
  // checks that the mutex is owned.
  void CheckWriteLocked() const CHECK_LOCKED() {
    CHECK(atomic_load(&state_, memory_order_relaxed) & kWriterLock);
  }

  void CheckLocked() const CHECK_LOCKED() { CheckWriteLocked(); }

  void CheckReadLocked() const CHECK_LOCKED() {
    CHECK(atomic_load(&state_, memory_order_relaxed) & kReaderLockMask);
  }

 private:
  atomic_uint64_t state_ = {0};
  Semaphore writers_;
  Semaphore readers_;

  // The state has 3 counters:
  //  - number of readers holding the lock,
  //    if non zero, the mutex is read-locked
  //  - number of waiting readers,
  //    if not zero, the mutex is write-locked
  //  - number of waiting writers,
  //    if non zero, the mutex is read- or write-locked
  // And 2 flags:
  //  - writer lock
  //    if set, the mutex is write-locked
  //  - a writer is awake and spin-waiting
  //    the flag is used to prevent thundering herd problem
  //    (new writers are not woken if this flag is set)
  //
  // Writer support active spinning, readers does not.
  // But readers are more aggressive and always take the mutex
  // if there are any other readers.
  // Writers hand off the mutex to readers: after wake up readers
  // already assume ownership of the mutex (don't need to do any
  // state updates). But the mutex is not handed off to writers,
  // after wake up writers compete to lock the mutex again.
  // This is needed to allow repeated write locks even in presence
  // of other blocked writers.
  static constexpr u64 kCounterWidth = 20;
  static constexpr u64 kReaderLockShift = 0;
  static constexpr u64 kReaderLockInc = 1ull << kReaderLockShift;
  static constexpr u64 kReaderLockMask = ((1ull << kCounterWidth) - 1)
                                         << kReaderLockShift;
  static constexpr u64 kWaitingReaderShift = kCounterWidth;
  static constexpr u64 kWaitingReaderInc = 1ull << kWaitingReaderShift;
  static constexpr u64 kWaitingReaderMask = ((1ull << kCounterWidth) - 1)
                                            << kWaitingReaderShift;
  static constexpr u64 kWaitingWriterShift = 2 * kCounterWidth;
  static constexpr u64 kWaitingWriterInc = 1ull << kWaitingWriterShift;
  static constexpr u64 kWaitingWriterMask = ((1ull << kCounterWidth) - 1)
                                            << kWaitingWriterShift;
  static constexpr u64 kWriterLock = 1ull << (3 * kCounterWidth);
  static constexpr u64 kWriterSpinWait = 1ull << (3 * kCounterWidth + 1);

  Mutex2(const Mutex2 &) = delete;
  void operator=(const Mutex2 &) = delete;
};

void FutexWait(atomic_uint32_t *p, u32 cmp);
void FutexWake(atomic_uint32_t *p, u32 count);

class MUTEX BlockingMutex {
 public:
  explicit constexpr BlockingMutex(LinkerInitialized)
      : opaque_storage_ {0, }, owner_ {0} {}
  BlockingMutex();
  void Lock() ACQUIRE();
  void Unlock() RELEASE();

  // This function does not guarantee an explicit check that the calling thread
  // is the thread which owns the mutex. This behavior, while more strictly
  // correct, causes problems in cases like StopTheWorld, where a parent thread
  // owns the mutex but a child checks that it is locked. Rather than
  // maintaining complex state to work around those situations, the check only
  // checks that the mutex is owned, and assumes callers to be generally
  // well-behaved.
  void CheckLocked() const CHECK_LOCKED();

 private:
  // Solaris mutex_t has a member that requires 64-bit alignment.
  ALIGNED(8) uptr opaque_storage_[10];
  uptr owner_;  // for debugging
};

// Reader-writer spin mutex.
class MUTEX RWMutex {
 public:
  RWMutex() {
    atomic_store(&state_, kUnlocked, memory_order_relaxed);
  }

  ~RWMutex() {
    CHECK_EQ(atomic_load(&state_, memory_order_relaxed), kUnlocked);
  }

  void Lock() ACQUIRE() {
    u32 cmp = kUnlocked;
    if (atomic_compare_exchange_strong(&state_, &cmp, kWriteLock,
                                       memory_order_acquire))
      return;
    LockSlow();
  }

  void Unlock() RELEASE() {
    u32 prev = atomic_fetch_sub(&state_, kWriteLock, memory_order_release);
    DCHECK_NE(prev & kWriteLock, 0);
    (void)prev;
  }

  void ReadLock() ACQUIRE_SHARED() {
    u32 prev = atomic_fetch_add(&state_, kReadLock, memory_order_acquire);
    if ((prev & kWriteLock) == 0)
      return;
    ReadLockSlow();
  }

  void ReadUnlock() RELEASE_SHARED() {
    u32 prev = atomic_fetch_sub(&state_, kReadLock, memory_order_release);
    DCHECK_EQ(prev & kWriteLock, 0);
    DCHECK_GT(prev & ~kWriteLock, 0);
    (void)prev;
  }

  void CheckLocked() const CHECK_LOCKED() {
    CHECK_NE(atomic_load(&state_, memory_order_relaxed), kUnlocked);
  }

 private:
  atomic_uint32_t state_;

  enum {
    kUnlocked = 0,
    kWriteLock = 1,
    kReadLock = 2
  };

  void NOINLINE LockSlow() {
    for (int i = 0;; i++) {
      if (i < 10)
        proc_yield(10);
      else
        internal_sched_yield();
      u32 cmp = atomic_load(&state_, memory_order_relaxed);
      if (cmp == kUnlocked &&
          atomic_compare_exchange_weak(&state_, &cmp, kWriteLock,
                                       memory_order_acquire))
          return;
    }
  }

  void NOINLINE ReadLockSlow() {
    for (int i = 0;; i++) {
      if (i < 10)
        proc_yield(10);
      else
        internal_sched_yield();
      u32 prev = atomic_load(&state_, memory_order_acquire);
      if ((prev & kWriteLock) == 0)
        return;
    }
  }

  RWMutex(const RWMutex &) = delete;
  void operator=(const RWMutex &) = delete;
};

template <typename MutexType>
class SCOPED_LOCK GenericScopedLock {
 public:
  explicit GenericScopedLock(MutexType *mu) ACQUIRE(mu) : mu_(mu) {
    mu_->Lock();
  }

  ~GenericScopedLock() RELEASE() { mu_->Unlock(); }

 private:
  MutexType *mu_;

  GenericScopedLock(const GenericScopedLock &) = delete;
  void operator=(const GenericScopedLock &) = delete;
};

template <typename MutexType>
class SCOPED_LOCK GenericScopedReadLock {
 public:
  explicit GenericScopedReadLock(MutexType *mu) ACQUIRE(mu) : mu_(mu) {
    mu_->ReadLock();
  }

  ~GenericScopedReadLock() RELEASE() { mu_->ReadUnlock(); }

 private:
  MutexType *mu_;

  GenericScopedReadLock(const GenericScopedReadLock &) = delete;
  void operator=(const GenericScopedReadLock &) = delete;
};

typedef GenericScopedLock<StaticSpinMutex> SpinMutexLock;
typedef GenericScopedLock<BlockingMutex> BlockingMutexLock;
typedef GenericScopedLock<RWMutex> RWMutexLock;
typedef GenericScopedReadLock<RWMutex> RWMutexReadLock;

}  // namespace __sanitizer

#endif  // SANITIZER_MUTEX_H
