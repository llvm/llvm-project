//===--- Implementation of a Darwin mutex class ------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_MUTEX_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_MUTEX_H

#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex_common.h"
#include "src/__support/threads/sleep.h" // For sleep_briefly
#include "src/__support/time/linux/abs_timeout.h"

#include <mach/mach_init.h> // For mach_thread_self
#include <mach/mach_port.h> // For mach_port_t and MACH_PORT_NULL
#include <os/lock.h>        // For os_unfair_lock
#include <time.h>           // For clock_gettime

namespace LIBC_NAMESPACE_DECL {

// This file is an implementation of `LIBC_NAMESPACE::mutex` for Darwin-based
// platforms. It is a wrapper around `os_unfair_lock`, which is a low-level,
// high-performance locking primitive provided by the kernel.
//
// `os_unfair_lock` is a non-recursive, thread-owned lock that blocks waiters
// efficiently in the kernel. As the name implies, it is "unfair," meaning
// it does not guarantee the order in which waiting threads acquire the lock.
// This trade-off allows for higher performance in contended scenarios.
//
// The lock must be unlocked from the same thread that locked it. Attempting
// to unlock from a different thread will result in a runtime error.
//
// This implementation is suitable for simple critical sections where fairness
// and reentrancy are not concerns.

class Mutex final {
  os_unfair_lock_s lock_val = OS_UNFAIR_LOCK_INIT;
  mach_port_t owner = MACH_PORT_NULL;

  // API compatibility fields.
  unsigned char timed;
  unsigned char recursive;
  unsigned char robust;
  unsigned char pshared;

public:
  LIBC_INLINE constexpr Mutex(bool is_timed, bool is_recursive, bool is_robust,
                              bool is_pshared)
      : owner(MACH_PORT_NULL), timed(is_timed), recursive(is_recursive),
        robust(is_robust), pshared(is_pshared) {}

  LIBC_INLINE constexpr Mutex()
      : owner(MACH_PORT_NULL), timed(0), recursive(0), robust(0), pshared(0) {}

  LIBC_INLINE static MutexError init(Mutex *mutex, bool is_timed, bool is_recur,
                                     bool is_robust, bool is_pshared) {
    mutex->lock_val = OS_UNFAIR_LOCK_INIT;
    mutex->owner = MACH_PORT_NULL;
    mutex->timed = is_timed;
    mutex->recursive = is_recur;
    mutex->robust = is_robust;
    mutex->pshared = is_pshared;
    return MutexError::NONE;
  }

  LIBC_INLINE static MutexError destroy(Mutex *lock) {
    LIBC_ASSERT(lock->owner == MACH_PORT_NULL &&
                "Mutex destroyed while locked.");
    return MutexError::NONE;
  }

  LIBC_INLINE MutexError lock() {
    os_unfair_lock_lock(&lock_val);
    owner = mach_thread_self();
    return MutexError::NONE;
  }

  LIBC_INLINE MutexError timed_lock(internal::AbsTimeout abs_time) {
    while (true) {
      if (try_lock() == MutexError::NONE) {
        return MutexError::NONE;
      }

      // Manually check if the timeout has expired.
      struct timespec now;
      // The clock used here must match the clock used to create the
      // absolute timeout.
      clock_gettime(abs_time.is_realtime() ? CLOCK_REALTIME : CLOCK_MONOTONIC,
                    &now);
      const timespec &target_ts = abs_time.get_timespec();

      if (now.tv_sec > target_ts.tv_sec || (now.tv_sec == target_ts.tv_sec &&
                                            now.tv_nsec >= target_ts.tv_nsec)) {
        // We might have acquired the lock between the last try_lock() and now.
        // To avoid returning TIMEOUT incorrectly, we do one last try_lock().
        if (try_lock() == MutexError::NONE)
          return MutexError::NONE;
        return MutexError::TIMEOUT;
      }

      sleep_briefly();
    }
  }

  LIBC_INLINE MutexError unlock() {
    // This check is crucial. It prevents both double-unlocks and unlocks
    // by threads that do not own the mutex.
    if (owner != mach_thread_self()) {
      return MutexError::UNLOCK_WITHOUT_LOCK;
    }
    owner = MACH_PORT_NULL;
    os_unfair_lock_unlock(&lock_val);
    return MutexError::NONE;
  }

  LIBC_INLINE MutexError try_lock() {
    if (os_unfair_lock_trylock(&lock_val)) {
      owner = mach_thread_self();
      return MutexError::NONE;
    }
    return MutexError::BUSY;
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_DARWIN_MUTEX_H
