//===--- Implementation of a Linux mutex class ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREAD_LINUX_MUTEX_H
#define LLVM_LIBC_SRC_SUPPORT_THREAD_LINUX_MUTEX_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/threads/mutex_common.h"

#include <linux/futex.h>
#include <stdint.h>
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

struct Mutex {
  unsigned char timed;
  unsigned char recursive;
  unsigned char robust;

  void *owner;
  unsigned long long lock_count;

  using FutexWordType = unsigned int;

  cpp::Atomic<FutexWordType> futex_word;

  enum class LockState : FutexWordType {
    Free,
    Locked,
    Waiting,
  };

public:
  constexpr Mutex(bool istimed, bool isrecursive, bool isrobust)
      : timed(istimed), recursive(isrecursive), robust(isrobust),
        owner(nullptr), lock_count(0),
        futex_word(FutexWordType(LockState::Free)) {}

  static MutexError init(Mutex *mutex, bool istimed, bool isrecur,
                         bool isrobust) {
    mutex->timed = istimed;
    mutex->recursive = isrecur;
    mutex->robust = isrobust;
    mutex->owner = nullptr;
    mutex->lock_count = 0;
    mutex->futex_word.set(FutexWordType(LockState::Free));
    return MutexError::NONE;
  }

  static MutexError destroy(Mutex *) { return MutexError::NONE; }

  MutexError reset();

  MutexError lock() {
    bool was_waiting = false;
    while (true) {
      FutexWordType mutex_status = FutexWordType(LockState::Free);
      FutexWordType locked_status = FutexWordType(LockState::Locked);

      if (futex_word.compare_exchange_strong(
              mutex_status, FutexWordType(LockState::Locked))) {
        if (was_waiting)
          futex_word = FutexWordType(LockState::Waiting);
        return MutexError::NONE;
      }

      switch (LockState(mutex_status)) {
      case LockState::Waiting:
        // If other threads are waiting already, then join them. Note that the
        // futex syscall will block if the futex data is still
        // `LockState::Waiting` (the 4th argument to the syscall function
        // below.)
        __llvm_libc::syscall(SYS_futex, &futex_word.val, FUTEX_WAIT_PRIVATE,
                             FutexWordType(LockState::Waiting), 0, 0, 0);
        was_waiting = true;
        // Once woken up/unblocked, try everything all over.
        continue;
      case LockState::Locked:
        // Mutex has been locked by another thread so set the status to
        // LockState::Waiting.
        if (futex_word.compare_exchange_strong(
                locked_status, FutexWordType(LockState::Waiting))) {
          // If we are able to set the futex data to `LockState::Waiting`, then
          // we will wait for the futex to be woken up. Note again that the
          // following syscall will block only if the futex data is still
          // `LockState::Waiting`.
          __llvm_libc::syscall(SYS_futex, &futex_word, FUTEX_WAIT_PRIVATE,
                               FutexWordType(LockState::Waiting), 0, 0, 0);
          was_waiting = true;
        }
        continue;
      case LockState::Free:
        // If it was LockState::Free, we shouldn't be here at all.
        return MutexError::BAD_LOCK_STATE;
      }
    }
  }

  MutexError unlock() {
    while (true) {
      FutexWordType mutex_status = FutexWordType(LockState::Waiting);
      if (futex_word.compare_exchange_strong(mutex_status,
                                             FutexWordType(LockState::Free))) {
        // If any thread is waiting to be woken up, then do it.
        __llvm_libc::syscall(SYS_futex, &futex_word, FUTEX_WAKE_PRIVATE, 1, 0,
                             0, 0);
        return MutexError::NONE;
      }

      if (mutex_status == FutexWordType(LockState::Locked)) {
        // If nobody was waiting at this point, just free it.
        if (futex_word.compare_exchange_strong(mutex_status,
                                               FutexWordType(LockState::Free)))
          return MutexError::NONE;
      } else {
        // This can happen, for example if some thread tries to unlock an
        // already free mutex.
        return MutexError::UNLOCK_WITHOUT_LOCK;
      }
    }
  }

  MutexError trylock();
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREAD_LINUX_MUTEX_H
