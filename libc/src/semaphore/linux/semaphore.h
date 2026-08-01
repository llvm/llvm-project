//===-- Linux Semaphore implementation for POSIX semaphores ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H
#define LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H

#include "hdr/errno_macros.h"
#include "hdr/time_macros.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/mode_t.h"
#include "hdr/types/struct_timespec.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/libc_assert.h"
#include "src/__support/threads/futex_utils.h"
#include "src/__support/time/abs_timeout.h"

namespace LIBC_NAMESPACE_DECL {

// Define SEM_VALUE_MAX as INT_MAX
constexpr unsigned int SEM_VALUE_MAX =
    static_cast<unsigned int>(cpp::numeric_limits<int>::max());

class Semaphore {
  Futex value;
  unsigned int canary;

  // Whether the semaphore is shared between processes.
  bool is_shared;

  // A private constant canary used to detect use of uninitialized or
  // destroyed semaphores. Chose "SEM1" in ASCII (0x53='S', 0x45='E',
  // 0x4D='M', 0x31='1').
  static constexpr unsigned int SEM_CANARY = 0x53454D31U;

  // Helper function for timed blocking wait, wait for a required
  // absolute timeout.
  LIBC_INLINE int wait_until(internal::AbsTimeout timeout) {
    while (trywait() != 0) {
      auto wait_or = value.wait(/*expected=*/0, timeout, is_shared);

      // returned error is not a benign race EAGAIN
      if (!wait_or.has_value() && wait_or.error() != EAGAIN) {

        // ETIMEDOUT still needs a final trywait() in case a post() raced with
        // the deadline
        if (wait_or.error() == ETIMEDOUT && trywait() == 0)
          return 0;

        // Any other error (e.g. EINVAL/EFAULT)
        return wait_or.error();
      }
    }
    return 0;
  }

public:
  LIBC_INLINE constexpr Semaphore(unsigned int value, bool shared)
      : value(value), canary(SEM_CANARY), is_shared(shared) {}

  // Sanity check to detect use of uninitialized or destroyed semaphores.
  LIBC_INLINE bool is_valid() const { return canary == SEM_CANARY; }

  LIBC_INLINE void destroy() {
    // Destroying a semaphore while threads are blocked on it is undefined
    // behavior. Similarly, using a destroyed semaphore is undefined.
    // Therefore no concurrency safe destruction is required here,
    // RELAXED memory ordering is sufficient.
    value.store(0, cpp::MemoryOrder::RELAXED);
    canary = 0;
  }

  LIBC_INLINE int getvalue() const {
    // get value is informational, not a synchronization op.
    // RELAXED ordering is enough.
    // TODO: handle the case where the semaphore is locked.
    return static_cast<int>(
        const_cast<Futex &>(value).load(cpp::MemoryOrder::RELAXED));
  }

  // Atomically increments the semaphore value and
  // wakes one blocked waiter if any.
  LIBC_INLINE int post() {
    FutexWordType v = value.load(cpp::MemoryOrder::RELAXED);

    do {
      if (v >= SEM_VALUE_MAX)
        return EOVERFLOW;
      // RELEASE on success, since post should publish prior writes
      // RELAXED on failure, since no synchronization event occurs
    } while (!value.compare_exchange_weak(v, v + 1, cpp::MemoryOrder::RELEASE,
                                          cpp::MemoryOrder::RELAXED));

    // Wake one waiter if any,
    // waiter selection is left to linux futex implementation.
    value.notify_one(is_shared);
    return 0;
  }

  // Attempts to decrement the semaphore value without blocking.
  LIBC_INLINE int trywait() {
    FutexWordType v = value.load(cpp::MemoryOrder::RELAXED);

    while (v > 0) {
      // ACQUIRE on success to synchronize with the matching post().
      if (value.compare_exchange_weak(v, v - 1, cpp::MemoryOrder::ACQUIRE,
                                      cpp::MemoryOrder::RELAXED))
        return 0;
    }

    return EAGAIN;
  }

  // Blocking wait, decrements the value when positive, or blocks until a
  // post() makes it positive again.
  LIBC_INLINE int wait() {
    while (trywait() != 0) {
      // Blocks when value was zero.
      // Futex wait re-checks the value atomically,
      // so a racing post() before sleep is not lost.
      // Spurious wakeups just send back to trywait().
      auto wait_or =
          value.wait(/*expected=*/0, /*timeout=*/cpp::nullopt, is_shared);

      // A successful wake or EAGAIN both just send it back to trywait().
      // Any other error is unexpected.
      if (!wait_or.has_value() && wait_or.error() != EAGAIN)
        return wait_or.error();
    }
    return 0;
  }

  // Blocking wait against an absolute CLOCK_REALTIME deadline.
  LIBC_INLINE int timedwait(const timespec *abstime) {
    return clockwait(CLOCK_REALTIME, abstime);
  }

  // Blocking wait against a deadline on the given clock:
  // CLOCK_REALTIME or CLOCK_MONOTONIC
  LIBC_INLINE int clockwait(clockid_t clock_id, const timespec *abstime) {
    if (clock_id != CLOCK_MONOTONIC && clock_id != CLOCK_REALTIME)
      return EINVAL;

    // If the semaphore can be locked immediately, succeed without consulting
    // the clock or dereferencing abstime.
    if (trywait() == 0)
      return 0;

    bool is_realtime = (clock_id == CLOCK_REALTIME);
    auto timeout =
        internal::AbsTimeout::from_timespec(*abstime, /*realtime=*/is_realtime);
    if (LIBC_LIKELY(timeout.has_value()))
      return wait_until(timeout.value());

    internal::AbsTimeout::Error err = timeout.error();
    LIBC_ASSERT(err == internal::AbsTimeout::Error::Invalid ||
                err == internal::AbsTimeout::Error::BeforeEpoch);
    switch (err) {
    case internal::AbsTimeout::Error::Invalid:
      return EINVAL;
    case internal::AbsTimeout::Error::BeforeEpoch:
      return ETIMEDOUT;
    }

    __builtin_trap();
  }

  // Named semaphore operations.
  // creates or opens a named semaphore backed by a file in /dev/shm/.
  // When O_CREAT is specified in oflag, mode and value are used for
  // initialization.
  static ErrorOr<Semaphore *> open(const char *name, int oflag, mode_t mode,
                                   unsigned int value);

  // unmaps a named semaphore.
  static int close(Semaphore *sem);

  // removes a named semaphore from the filesystem.
  static int unlink(const char *name);
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H
