//===-- Utility mutex classes -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_LINUX_MUTEX_H
#define LLVM_LIBC_SRC_THREADS_LINUX_MUTEX_H

#include "Futex.h"

#include "config/linux/syscall.h" // For syscall functions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For values like thrd_success etc.

#include <linux/futex.h> // For futex operations.
#include <stdatomic.h>
#include <stdint.h>

namespace __llvm_libc {

struct Mutex {
  enum Status : uint32_t {
    MS_Free,
    MS_Locked,
    MS_Waiting,
  };

  FutexWord futex_word;
  int type;

  static int init(Mutex *mutex, int type) {
    mutex->futex_word = MS_Free;
    mutex->type = type;
    return thrd_success;
  }

  int lock() {
    bool was_waiting = false;
    while (true) {
      uint32_t mutex_status = MS_Free;
      uint32_t locked_status = MS_Locked;

      if (atomic_compare_exchange_strong(&futex_word, &mutex_status,
                                         MS_Locked)) {
        if (was_waiting)
          atomic_store(&futex_word, MS_Waiting);
        return thrd_success;
      }

      switch (mutex_status) {
      case MS_Waiting:
        // If other threads are waiting already, then join them. Note that the
        // futex syscall will block if the futex data is still `MS_Waiting` (the
        // 4th argument to the syscall function below.)
        __llvm_libc::syscall(SYS_futex, &futex_word, FUTEX_WAIT_PRIVATE,
                             MS_Waiting, 0, 0, 0);
        was_waiting = true;
        // Once woken up/unblocked, try everything all over.
        continue;
      case MS_Locked:
        // Mutex has been locked by another thread so set the status to
        // MS_Waiting.
        if (atomic_compare_exchange_strong(&futex_word, &locked_status,
                                           MS_Waiting)) {
          // If we are able to set the futex data to `MS_Waiting`, then we will
          // wait for the futex to be woken up. Note again that the following
          // syscall will block only if the futex data is still `MS_Waiting`.
          __llvm_libc::syscall(SYS_futex, &futex_word, FUTEX_WAIT_PRIVATE,
                               MS_Waiting, 0, 0, 0);
          was_waiting = true;
        }
        continue;
      case MS_Free:
        // If it was MS_Free, we shouldn't be here at all.
        [[clang::fallthrough]];
      default:
        // Mutex status cannot be anything else. So control should not reach
        // here at all.
        return thrd_error;
      }
    }
  }

  int unlock() {
    while (true) {
      uint32_t mutex_status = MS_Waiting;
      if (atomic_compare_exchange_strong(&futex_word, &mutex_status, MS_Free)) {
        // If any thread is waiting to be woken up, then do it.
        __llvm_libc::syscall(SYS_futex, &futex_word, FUTEX_WAKE_PRIVATE, 1, 0,
                             0, 0);
        return thrd_success;
      }

      if (mutex_status == MS_Locked) {
        // If nobody was waiting at this point, just free it.
        if (atomic_compare_exchange_strong(&futex_word, &mutex_status, MS_Free))
          return thrd_success;
      } else {
        // This can happen, for example if some thread tries to unlock an
        // already free mutex.
        return thrd_error;
      }
    }
  }
};

static_assert(sizeof(Mutex) == sizeof(mtx_t),
              "Sizes of internal representation of mutex and the public mtx_t "
              "do not match.");

class MutexLock {
  Mutex *mutex;

public:
  explicit MutexLock(Mutex *m) : mutex(m) { mutex->lock(); }

  ~MutexLock() { mutex->unlock(); }
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_LINUX_MUTEX_H
