//===-- Utility condition variable class ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/threads/CndVar.h"
#include "src/__support/CPP/mutex.h"
#include "src/__support/OSUtil/syscall.h" // syscall_impl
#include "src/__support/macros/config.h"
#include "src/__support/threads/linux/futex_word.h" // FutexWordType
#include "src/__support/threads/mutex.h"            // Mutex
#include "src/__support/threads/raw_mutex.h"        // RawMutex
#include "src/__support/time/monotonicity.h"
#include <sys/syscall.h> // For syscall numbers.

#ifndef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#define LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY 1
#endif

namespace LIBC_NAMESPACE_DECL {

CndVar::Result CndVar::wait(Mutex *m,
                            cpp::optional<internal::AbsTimeout> timeout) {
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
    push_back(&waiter);

    if (m->unlock() != MutexError::NONE) {
      // If we do not remove the queued up waiter before returning,
      // then another thread can potentially signal a non-existing
      // waiter. Note also that we do this with |qmtx| locked. This
      // ensures that another thread will not signal the withdrawing
      // waiter.
      remove(&waiter);

      return Result::MutexError;
    }
  }
#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
  if (timeout.has_value())
    internal::ensure_monotonicity(*timeout);
#endif
  if (waiter.futex_word.wait(WS_Waiting, timeout, true) == -ETIMEDOUT) {
    cpp::lock_guard ml(qmtx);
    remove(&waiter);
    return Result::Timeout;
  }

  // At this point, if locking |m| fails, we can simply return as the
  // queued up waiter would have been removed from the queue.
  auto err = m->lock();
  return err == MutexError::NONE ? Result::Success : Result::MutexError;
}

void CndVar::notify_one() {
  // We don't use an RAII locker in this method as we want to unlock
  // |qmtx| and signal the waiter using a single FUTEX_WAKE_OP signal.
  qmtx.lock();
  CndWaiter *first = pop_front();
  if (first == nullptr)
    qmtx.unlock();

  qmtx.reset();

  // this is a special WAKE_OP, so we use syscall directly
  LIBC_NAMESPACE::syscall_impl<long>(
      FUTEX_SYSCALL_ID, &qmtx.get_raw_futex(), FUTEX_WAKE_OP, 1, 1,
      &first->futex_word.val,
      FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
}

void CndVar::broadcast() {
  // needs to hold until broadcast is done to avoid timeout race condition
  cpp::lock_guard ml(qmtx);
  CndWaiter *waiter = take_all();
  uint32_t dummy_futex_word;
  while (waiter != nullptr) {
    // FUTEX_WAKE_OP is used instead of just FUTEX_WAKE as it allows us to
    // atomically update the waiter status to WS_Signalled before waking
    // up the waiter. A dummy location is used for the other futex of
    // FUTEX_WAKE_OP.
    LIBC_NAMESPACE::syscall_impl<long>(
        FUTEX_SYSCALL_ID, &dummy_futex_word, FUTEX_WAKE_OP, 1, 1,
        &waiter->futex_word.val,
        FUTEX_OP(FUTEX_OP_SET, WS_Signalled, FUTEX_OP_CMP_EQ, WS_Waiting));
    waiter = static_cast<CndWaiter *>(waiter->next);
  }
}

void CndVar::ensure_cyclic_queue() {
  if (waitq.prev || waitq.next)
    return;
  waitq.prev = waitq.next = &waitq;
}

void CndVar::push_back(CndWaiter *w) {
  ensure_cyclic_queue();
  w->next = &waitq;
  w->prev = waitq.prev;
  w->next->prev = w;
  w->prev->next = w;
}

CndVar::CndWaiter *CndVar::pop_front() {
  ensure_cyclic_queue();
  if (waitq.next == &waitq)
    return nullptr;
  CndWaiter *first = static_cast<CndWaiter *>(waitq.next);
  remove(first);
  return first;
}

CndVar::CndWaiter *CndVar::take_all() {
  ensure_cyclic_queue();
  if (waitq.next == &waitq)
    return nullptr;
  waitq.next->prev = nullptr;
  waitq.prev->next = nullptr;
  CndVar::CndWaiter *first = static_cast<CndWaiter *>(waitq.next);
  waitq.next = waitq.prev = &waitq;
  return first;
}

void CndVar::remove(CndWaiter *w) {
  // node may have already been removed from the queue
  if (w->next == nullptr || w->prev == nullptr)
    return;
  w->next->prev = w->prev;
  w->prev->next = w->next;
  w->next = w->prev = nullptr;
}

} // namespace LIBC_NAMESPACE_DECL
