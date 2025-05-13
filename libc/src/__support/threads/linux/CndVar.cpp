//===-- Utility condition variable class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/CndVar.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/syscall.h"           // syscall_impl
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/futex_word.h" // FutexWordType
#include "src/__support/threads/linux/raw_mutex.h"  // RawMutex
#include "src/__support/threads/mutex.h"            // Mutex

#include <sys/syscall.h> // For syscall numbers.

namespace LIBC_NAMESPACE_DECL {

int CndVar::wait(Mutex *m) {
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
    cpp::lock_guard ml(qmtx);
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

      return -1;
    }
  }

  waiter.futex_word.wait(WS_Waiting, cpp::nullopt, true);

  // At this point, if locking |m| fails, we can simply return as the
  // queued up waiter would have been removed from the queue.
  auto err = m->lock();
  return err == MutexError::NONE ? 0 : -1;
}

void CndVar::notify_one() {
  // We don't use an RAII locker in this method as we want to unlock
  // |qmtx| and signal the waiter using a single FUTEX_WAKE_OP signal.
  qmtx.lock();
  if (waitq_front == nullptr)
    qmtx.unlock();

  CndWaiter *first = waitq_front;
  waitq_front = waitq_front->next;
  if (waitq_front == nullptr)
    waitq_back = nullptr;

  qmtx.reset();

  // this is a special WAKE_OP, so we use syscall directly
  LIBC_NAMESPACE::syscall_impl<long>(
      FUTEX_SYSCALL_ID, &qmtx.get_raw_futex(), FUTEX_WAKE_OP, 1, 1,
      &first->futex_word.val,
      FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
}

void CndVar::broadcast() {
  cpp::lock_guard ml(qmtx);
  uint32_t dummy_futex_word;
  CndWaiter *waiter = waitq_front;
  waitq_front = waitq_back = nullptr;
  while (waiter != nullptr) {
    // FUTEX_WAKE_OP is used instead of just FUTEX_WAKE as it allows us to
    // atomically update the waiter status to WS_Signalled before waking
    // up the waiter. A dummy location is used for the other futex of
    // FUTEX_WAKE_OP.
    LIBC_NAMESPACE::syscall_impl<long>(
        FUTEX_SYSCALL_ID, &dummy_futex_word, FUTEX_WAKE_OP, 1, 1,
        &waiter->futex_word.val,
        FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
    waiter = waiter->next;
  }
}

} // namespace LIBC_NAMESPACE_DECL
