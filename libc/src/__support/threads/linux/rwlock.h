//===--- Implementation of a Linux rwlock class ------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_MUTEX_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/threads/linux/futex_word.h"

#include "src/__support/threads/sleep.h"
#include "src/errno/libc_errno.h"
#include "src/time/clock_gettime.h"

namespace LIBC_NAMESPACE {
// A rwlock implementation using futexes.
// Code is largely based on the Rust standard library implementation.
// https://github.com/rust-lang/rust/blob/22a5267c83a3e17f2b763279eb24bb632c45dc6b/library/std/src/sys/sync/rwlock/futex.rs
struct RwLock {
  // The state consists of a 30-bit reader counter, a 'readers waiting' flag,
  // and a 'writers waiting' flag. Bits 0..30:
  //   0: Unlocked
  //   1..=0x3FFF_FFFE: Locked by N readers
  //   0x3FFF_FFFF: Write locked
  // Bit 30: Readers are waiting on this futex.
  // Bit 31: Writers are waiting on the writer_notify futex.
  cpp::Atomic<FutexWordType> state;
  // The 'condition variable' to notify writers through.
  // Incremented on every signal.
  cpp::Atomic<FutexWordType> writer_notify;
  // If the rwlock is shared between processes.
  bool is_shared;

  LIBC_INLINE_VAR static constexpr FutexWordType READ_LOCKED = 1u;
  LIBC_INLINE_VAR static constexpr FutexWordType MASK = (1u << 30) - 1u;
  LIBC_INLINE_VAR static constexpr FutexWordType WRITE_LOCKED = MASK;
  LIBC_INLINE_VAR static constexpr FutexWordType MAX_READERS = MASK - 1u;
  LIBC_INLINE_VAR static constexpr FutexWordType READERS_WAITING = 1u << 30u;
  LIBC_INLINE_VAR static constexpr FutexWordType WRITERS_WAITING = 1u << 31u;
  LIBC_INLINE_VAR static constexpr int_fast32_t SPIN_LIMIT = 100u;
  LIBC_INLINE static constexpr bool is_unlocked(FutexWordType state) {
    return (state & MASK) == 0u;
  }
  LIBC_INLINE static constexpr bool is_write_locked(FutexWordType state) {
    return (state & MASK) == WRITE_LOCKED;
  }
  LIBC_INLINE static constexpr bool has_readers_waiting(FutexWordType state) {
    return (state & READERS_WAITING) != 0u;
  }
  LIBC_INLINE static constexpr bool has_writers_waiting(FutexWordType state) {
    return (state & WRITERS_WAITING) != 0u;
  }
  // This also returns false if the counter could overflow if we tried to read
  // lock it.
  //
  // We don't allow read-locking if there's readers waiting, even if the lock is
  // unlocked and there's no writers waiting. The only situation when this
  // happens is after unlocking, at which point the unlocking thread might be
  // waking up writers, which have priority over readers. The unlocking thread
  // will clear the readers waiting bit and wake up readers, if necessary.
  LIBC_INLINE static constexpr bool is_read_lockable(FutexWordType state) {
    return (state & MASK) < MAX_READERS && !has_readers_waiting(state) &&
           !has_writers_waiting(state);
  }
  LIBC_INLINE static constexpr bool
  has_reached_max_readers(FutexWordType state) {
    return (state & MASK) == MAX_READERS;
  }

  // Convert a relative timeout to an absolute timespec.
  LIBC_INLINE static void abs_timeout(cpp::optional<timespec> &timeout) {
    if (!timeout)
      return;

    int errno_backup = libc_errno;
    timespec now;
    // if we failed to get time, we move timeout to infinity by setting it to
    // nullopt.
    if (LIBC_NAMESPACE::clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
      timeout = cpp::nullopt;
      libc_errno = errno_backup;
      return;
    }
    if (__builtin_add_overflow(now.tv_sec, timeout->tv_sec, &timeout->tv_sec)) {
      timeout = cpp::nullopt;
      return;
    }
    if (__builtin_add_overflow(now.tv_nsec, timeout->tv_nsec,
                               &timeout->tv_nsec)) {
      timeout = cpp::nullopt;
      return;
    }
    if (now.tv_nsec >= 1'000'000'000) {
      if (__builtin_add_overflow(now.tv_sec, 1, &timeout->tv_sec)) {
        timeout = cpp::nullopt;
        return;
      }
      timeout->tv_nsec -= 1'000'000'000;
    }
  }

  template <typename F>
  LIBC_INLINE bool fetch_update(cpp::Atomic<FutexWordType> &__restrict word,
                                FutexWordType &__restrict prev,
                                cpp::MemoryOrder set_order,
                                cpp::MemoryOrder fetch_order, F &&f) {
    prev = word.load(fetch_order);
    // It is okay to have spurious failures here as we are in a loop.
    while (cpp::optional<FutexWordType> new_val = f(prev))
      if (word.compare_exchange_weak(prev, *new_val, set_order, fetch_order))
        return true;

    return false;
  }

  template <typename F>
  LIBC_INLINE FutexWordType spin_until(int_fast32_t spin, F &&f) {
    for (;;) {
      FutexWordType state = this->state.load(cpp::MemoryOrder::RELAXED);
      if (f(state) || spin == 0)
        return state;
      // pause the pipeline to avoid extra memory loads due to speculation
      sleep_briefly();
      --spin;
    }
  }

  LIBC_INLINE static bool
  is_timeout(const cpp::optional<timespec> &abs_timeout) {
    if (!abs_timeout)
      return false;
    timespec now;
    LIBC_NAMESPACE::clock_gettime(CLOCK_MONOTONIC, &now);
    if (now.tv_sec > abs_timeout->tv_sec ||
        (now.tv_sec == abs_timeout->tv_sec &&
         now.tv_nsec >= abs_timeout->tv_nsec))
      return true;
    return false;
  }

  LIBC_INLINE FutexWordType spin_read(int_fast32_t spin) {
    // Stop spinning when it's unlocked or read locked, or when there's waiting
    // threads.
    return spin_until(spin, [&](FutexWordType state) -> bool {
      return !is_write_locked(state) || has_readers_waiting(state) ||
             has_writers_waiting(state) || is_unlocked(state);
    });
  }

  LIBC_INLINE FutexWordType spin_write(int_fast32_t spin) {
    // Stop spinning when it's unlocked or when there's waiting writers, to keep
    // things somewhat fair.
    return spin_until(spin, [&](FutexWordType state) -> bool {
      return is_unlocked(state) || has_writers_waiting(state);
    });
  }

  [[gnu::cold]]
  LIBC_INLINE bool
  read_contented(cpp::optional<timespec> abs_timeout = cpp::nullopt) {
    // Notice that the timeout is not checked during the fast spin loop.
    FutexWordType prev = this->spin_read(SPIN_LIMIT);
    int lockable_loop = 0;
    for (;;) {
      // We have a chance to lock it, go ahead and have a try.
      if (is_read_lockable(prev)) {
        if (this->state.compare_exchange_weak(prev, prev + READ_LOCKED,
                                              cpp::MemoryOrder::ACQUIRE,
                                              cpp::MemoryOrder::RELAXED))
          return true;
        // Timeout is checked the first time we enter the loop since there is
        // always spin before this.
        if (lockable_loop == 0 && is_timeout(abs_timeout)) {
          libc_errno = ETIMEDOUT;
          return false;
        }
        // In the rare case, we somehow stuck in the lockable loop for a long
        // time. We should give a chance to allow timeout checking. We do this
        // everytime we meet the SPIN_LIMIT.
        lockable_loop = (lockable_loop + 1) % SPIN_LIMIT;
        // we continue to spin if we failed to lock it.
        // notice that prev is updated by the compare_exchange_weak.
        continue;
      }

      if (has_reached_max_readers(prev)) {
        // The read lock could not be acquired because the maximum
        // number of read locks for rwlock has been exceeded.
        libc_errno = EAGAIN;
        return false;
      }

      // Make sure the readers waiting bit is set before we go to sleep.
      // Strong CAS is required as this is a one-time operation.
      if (!has_readers_waiting(prev) &&
          !this->state.compare_exchange_strong(prev, prev | READERS_WAITING,
                                               cpp::MemoryOrder::RELAXED))
        continue;

      if (!futex_wait(this->state, prev | READERS_WAITING, abs_timeout,
                      is_shared)) {
        // timeout happened.
        libc_errno = ETIMEDOUT;
        return false;
      }

      prev = this->spin_read(SPIN_LIMIT);
      lockable_loop = 0;
    }
  }

  /// Wake up a single writer.
  LIBC_INLINE bool wake_writer() {
    writer_notify.fetch_add(1, cpp::MemoryOrder::RELEASE);
    return futex_wake_one(writer_notify);
  }

  /// Wake up waiting threads after unlocking.
  ///
  /// If both are waiting, this will wake up only one writer, but will fall
  /// back to waking up readers if there was no writer to wake up.
  [[gnu::cold]]
  LIBC_INLINE void wake_writer_or_readers(FutexWordType prev) {
    LIBC_ASSERT(is_unlocked(prev));
    // The readers waiting bit might be turned on at any point now,
    // since readers will block when there's anything waiting.
    // Writers will just lock the lock though, regardless of the waiting bits,
    // so we don't have to worry about the writer waiting bit.
    //
    // If the lock gets locked in the meantime, we don't have to do
    // anything, because then the thread that locked the lock will take
    // care of waking up waiters when it unlocks.

    // If only writers are waiting, wake one of them up.
    if (prev == WRITERS_WAITING)
      if (this->state.compare_exchange_strong(
              prev, 0, cpp::MemoryOrder::RELAXED, cpp::MemoryOrder::RELAXED))
        return (void)wake_writer();
    // notice that prev is updated.

    // If both writers and readers are waiting, leave the readers waiting
    // and only wake up one writer.
    if (prev == (WRITERS_WAITING + READERS_WAITING)) {
      // set READERS_WAITING and check if someone else locked the lock.
      if (!this->state.compare_exchange_strong(prev, READERS_WAITING,
                                               cpp::MemoryOrder::RELAXED,
                                               cpp::MemoryOrder::RELAXED))
        return; // someone else locked the lock.

      if (wake_writer())
        return;

      // No writers were actually blocked on futex_wait, so we continue
      // to wake up readers instead, since we can't be sure if we notified a
      // writer.
      prev = READERS_WAITING;
    }

    if (prev == READERS_WAITING &&
        this->state.compare_exchange_strong(prev, 0, cpp::MemoryOrder::RELAXED,
                                            cpp::MemoryOrder::RELAXED))
      futex_wake_all(this->state);
  }

  [[gnu::cold]]
  LIBC_INLINE bool write_contented(cpp::optional<timespec> abs_timeout) {
    FutexWordType prev = this->spin_write(SPIN_LIMIT);
    FutexWordType other_writers_waiting = 0;
    int lockable_loop = 0;
    for (;;) {
      // We have a chance to lock it, go ahead and have a try.
      if (is_unlocked(prev)) {
        if (this->state.compare_exchange_weak(
                prev, prev | WRITE_LOCKED | other_writers_waiting,
                cpp::MemoryOrder::ACQUIRE, cpp::MemoryOrder::RELAXED))
          return true;
        // Timeout is checked the first time we enter the loop since there is
        // always spin before this.
        if (lockable_loop == 0 && is_timeout(abs_timeout)) {
          libc_errno = ETIMEDOUT;
          return false;
        }
        // In the rare case, we somehow stuck in the lockable loop for a long
        // time. We should give a chance to allow timeout checking. We do this
        // everytime we meet the SPIN_LIMIT.
        lockable_loop = (lockable_loop + 1) % SPIN_LIMIT;
        // we continue to spin if we failed to lock it.
        // notice that prev is updated by the compare_exchange_weak.
        continue;
      }

      // Set the waiting bit indicating that we're waiting on it.
      if (!has_writers_waiting(prev) &&
          !this->state.compare_exchange_strong(prev, prev | WRITERS_WAITING,
                                               cpp::MemoryOrder::RELAXED))
        continue;

      // Other writers might be waiting now too, so we should make sure
      // we keep that bit on once we manage lock it.
      other_writers_waiting = WRITERS_WAITING;

      // Examine the notification counter before we check if `state` has
      // changed, to make sure we don't miss any notifications.
      FutexWordType seq = this->writer_notify.load(cpp::MemoryOrder::ACQUIRE);

      // Don't go to sleep if the lock has become available,
      // or if the writers waiting bit is no longer set.
      prev = this->state.load(cpp::MemoryOrder::RELAXED);
      if (is_unlocked(prev) || !has_writers_waiting(prev))
        continue;

      // Wait for the state to change.
      if (!futex_wait(this->writer_notify, seq, abs_timeout, is_shared)) {
        // timeout happened.
        libc_errno = ETIMEDOUT;
        return false;
      }

      // Spin again after waking up.
      prev = this->spin_write(SPIN_LIMIT);
      lockable_loop = 0;
    }
  }

public:
  LIBC_INLINE explicit constexpr RwLock(bool shared = false)
      : state(FutexWordType(0)), writer_notify(FutexWordType(0)),
        is_shared(shared) {}

  LIBC_INLINE bool try_read() {
    FutexWordType prev;
    // set order is the success order for the inner compare_exchange_weak.
    // it effectively make the store ordering to be relaxed.
    return fetch_update(this->state, prev, cpp::MemoryOrder::ACQUIRE,
                        cpp::MemoryOrder::RELAXED,
                        [](FutexWordType x) -> cpp::optional<FutexWordType> {
                          if (is_read_lockable(x))
                            return {x + READ_LOCKED};
                          return cpp::nullopt;
                        });
  }

  LIBC_INLINE bool read(cpp::optional<timespec> timeout = cpp::nullopt) {
    // record absoulte time if timeout is provided.
    abs_timeout(timeout);
    FutexWordType prev = this->state.load(cpp::MemoryOrder::RELAXED);
    // It is okay to have spurious failures here as we will fallback to
    // contented routine.
    if (!is_read_lockable(prev) ||
        !this->state.compare_exchange_weak(prev, prev + READ_LOCKED,
                                           cpp::MemoryOrder::ACQUIRE,
                                           cpp::MemoryOrder::RELAXED))
      return read_contented(timeout);
    return true;
  }

  LIBC_INLINE void read_unlock() {
    FutexWordType prev =
        this->state.fetch_sub(READ_LOCKED, cpp::MemoryOrder::RELEASE) -
        READ_LOCKED;
    // It's impossible for a reader to be waiting on a read-locked RwLock,
    // except if there is also a writer waiting.
    LIBC_ASSERT(!has_readers_waiting(state) || has_writers_waiting(state));

    // Wake up a writer if we were the last reader and there's a writer waiting.
    if (is_unlocked(prev) && has_writers_waiting(prev))
      wake_writer_or_readers(prev);
  }
  LIBC_INLINE bool try_write() {
    FutexWordType prev;
    return fetch_update(this->state, prev, cpp::MemoryOrder::ACQUIRE,
                        cpp::MemoryOrder::RELAXED,
                        [](FutexWordType s) -> cpp::optional<FutexWordType> {
                          if (is_unlocked(s))
                            return s + WRITE_LOCKED;
                          return cpp::nullopt;
                        });
  }
  LIBC_INLINE bool write(cpp::optional<timespec> timeout = cpp::nullopt) {
    // record absoulte time if timeout is provided.
    abs_timeout(timeout);
    FutexWordType prev = 0;
    // It is okay to have spurious failures here as we will fallback to
    // contented routine.
    if (!this->state.compare_exchange_weak(prev, WRITE_LOCKED,
                                           cpp::MemoryOrder::ACQUIRE,
                                           cpp::MemoryOrder::RELAXED))
      return write_contented(timeout);
    return true;
  }
  LIBC_INLINE void write_unlock() {
    FutexWordType prev =
        this->state.fetch_sub(WRITE_LOCKED, cpp::MemoryOrder::RELEASE) -
        WRITE_LOCKED;

    LIBC_ASSERT(is_unlocked(prev));

    if (has_writers_waiting(prev) || has_readers_waiting(prev))
      wake_writer_or_readers(prev);
  }
};

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_LINUX_MUTEX_H
