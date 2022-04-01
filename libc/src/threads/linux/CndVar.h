//===-- Utility condition variable class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_THREADS_LINUX_CNDVAR_H
#define LLVM_LIBC_SRC_THREADS_LINUX_CNDVAR_H

#include "include/sys/syscall.h" // For syscall numbers.
#include "include/threads.h"     // For values like thrd_success etc.
#include "src/__support/CPP/atomic.h"
#include "src/__support/OSUtil/syscall.h" // For syscall functions.
#include "src/__support/threads/linux/futex_word.h"
#include "src/__support/threads/mutex.h"

#include <linux/futex.h> // For futex operations.
#include <stdint.h>

namespace __llvm_libc {

struct CndVar {
  enum CndWaiterStatus : uint32_t {
    WS_Waiting = 0xE,
    WS_Signalled = 0x5,
  };

  struct CndWaiter {
    cpp::Atomic<uint32_t> futex_word = WS_Waiting;
    CndWaiter *next = nullptr;
  };

  CndWaiter *waitq_front;
  CndWaiter *waitq_back;
  Mutex qmtx;

  static int init(CndVar *cv) {
    cv->waitq_front = cv->waitq_back = nullptr;
    auto err = Mutex::init(&cv->qmtx, false, false, false);
    return err == MutexError::NONE ? thrd_success : thrd_error;
  }

  static void destroy(CndVar *cv) {
    cv->waitq_front = cv->waitq_back = nullptr;
  }

  int wait(Mutex *m) {
    // The goal is to perform "unlock |m| and wait" in an
    // atomic operation. However, it is not possible to do it
    // in the true sense so we do it in spirit. Before unlocking
    // |m|, a new waiter object is added to the waiter queue with
    // the waiter queue locked. Iff a signalling thread signals
    // the waiter before the waiter actually starts waiting, the
    // wait operation will not begin at all and the waiter immediately
    // returns.

    CndWaiter waiter;
    {
      MutexLock ml(&qmtx);
      CndWaiter *old_back = nullptr;
      if (waitq_front == nullptr) {
        waitq_front = waitq_back = &waiter;
      } else {
        old_back = waitq_back;
        waitq_back->next = &waiter;
        waitq_back = &waiter;
      }

      if (m->unlock() != MutexError::NONE) {
        // If we do not remove the queued up waiter before returning,
        // then another thread can potentially signal a non-existing
        // waiter. Note also that we do this with |qmtx| locked. This
        // ensures that another thread will not signal the withdrawing
        // waiter.
        waitq_back = old_back;
        if (waitq_back == nullptr)
          waitq_front = nullptr;
        else
          waitq_back->next = nullptr;

        return thrd_error;
      }
    }

    __llvm_libc::syscall(SYS_futex, &waiter.futex_word.val, FUTEX_WAIT,
                         WS_Waiting, 0, 0, 0);

    // At this point, if locking |m| fails, we can simply return as the
    // queued up waiter would have been removed from the queue.
    auto err = m->lock();
    return err == MutexError::NONE ? thrd_success : thrd_error;
  }

  int notify_one() {
    // We don't use an RAII locker in this method as we want to unlock
    // |qmtx| and signal the waiter using a single FUTEX_WAKE_OP signal.
    qmtx.lock();
    if (waitq_front == nullptr) {
      qmtx.unlock();
      return thrd_success;
    }

    CndWaiter *first = waitq_front;
    waitq_front = waitq_front->next;
    if (waitq_front == nullptr)
      waitq_back = nullptr;

    qmtx.futex_word = FutexWordType(Mutex::LockState::Free);

    __llvm_libc::syscall(
        SYS_futex, &qmtx.futex_word.val, FUTEX_WAKE_OP, 1, 1,
        &first->futex_word.val,
        FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
    return thrd_success;
  }

  int broadcast() {
    MutexLock ml(&qmtx);
    uint32_t dummy_futex_word;
    CndWaiter *waiter = waitq_front;
    waitq_front = waitq_back = nullptr;
    while (waiter != nullptr) {
      // FUTEX_WAKE_OP is used instead of just FUTEX_WAKE as it allows us to
      // atomically update the waiter status to WS_Signalled before waking
      // up the waiter. A dummy location is used for the other futex of
      // FUTEX_WAKE_OP.
      __llvm_libc::syscall(
          SYS_futex, &dummy_futex_word, FUTEX_WAKE_OP, 1, 1,
          &waiter->futex_word.val,
          FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
      waiter = waiter->next;
    }
    return thrd_success;
  }
};

static_assert(sizeof(CndVar) == sizeof(cnd_t),
              "Mismatch in the size of the "
              "internal representation of condition variable and the public "
              "cnd_t type.");

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_THREADS_LINUX_CNDVAR_H
