//===-- A platform independent abstraction layer for cond vars --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_THREADS_CNDVAR_H
#define LLVM_LIBC_SRC___SUPPORT_THREADS_CNDVAR_H

#include "hdr/stdint_proxy.h" // uint32_t
#include "src/__support/CPP/mutex.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/futex_utils.h" // Futex
#include "src/__support/threads/mutex.h"       // Mutex
#include "src/__support/threads/raw_mutex.h"   // RawMutex

#ifndef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#define LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY 1
#endif

namespace LIBC_NAMESPACE_DECL {

class CndVar {
  enum WaiterState : uint8_t {
    // Initial state after entering the wait queue.
    Waiting = 0,
    // A signal has been received.
    Signalled = 1,
    // A cancellation has been requested.
    Cancelled = 2,
  };

  struct WaiterHeader {
    WaiterHeader *prev;
    WaiterHeader *next;

    LIBC_INLINE void ensure_queue_initialization() {
      if (LIBC_UNLIKELY(prev == nullptr)) {
        prev = next = this;
      }
    }

    LIBC_INLINE void push_back(WaiterHeader *waiter) {
      ensure_queue_initialization();
      waiter->next = this;
      waiter->prev = prev;
      waiter->next->prev = waiter;
      waiter->prev->next = waiter;
    }

    LIBC_INLINE static void remove(WaiterHeader *waiter) {
      waiter->next->prev = waiter->prev;
      waiter->prev->next = waiter->next;
      waiter->prev = waiter->next = waiter;
    }

    LIBC_INLINE WaiterHeader *pop_front() {
      ensure_queue_initialization();
      if (next == this)
        return nullptr;
      WaiterHeader *first = next;
      remove(first);
      return first;
    }
  };
  struct CndWaiter : WaiterHeader {
    cpp::Atomic<Futex *> sender_futex;
    RawMutex barrier;
    cpp::Atomic<uint8_t> state;

    LIBC_INLINE CndWaiter()
        : WaiterHeader{}, sender_futex(nullptr), barrier{}, state{Waiting} {
      // this lock should always success as no contention is possible
      (void)barrier.try_lock();
    }

    LIBC_INLINE void confirm_cancellation() {
      Futex *sender = sender_futex.load();
      if (sender && sender->fetch_sub(1) == 1)
        sender->notify_one();
    }
  };

  union {
    struct {
      RawMutex queue_lock;
      WaiterHeader waiter_queue;
    };
    struct {
      Futex shared_futex;
      cpp::Atomic<size_t> shared_waiters;
    };
  };

public:
  enum class CndVarResult {
    Success,
    MutexError,
    Timeout,
  };

  using Timeout = internal::AbsTimeout;

  LIBC_INLINE constexpr CndVar() : queue_lock{}, waiter_queue{} {}

  LIBC_INLINE void reset() {
    queue_lock.reset();
    waiter_queue.prev = nullptr;
    waiter_queue.next = nullptr;
  }

  // TODO: register callback for pthread cancellation
  LIBC_INLINE CndVarResult wait(Mutex *mutex,
                                cpp::optional<Timeout> timeout = cpp::nullopt,
                                bool is_shared = false) {
#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
    if (timeout)
      ensure_monotonicity(*timeout);
#endif

    if (is_shared) {
      shared_waiters.fetch_add(1);
      FutexWordType old_val = shared_futex.load();
      mutex->unlock();
      ErrorOr<int> result =
          shared_futex.wait(old_val, timeout, /*is_pshared=*/true);
      shared_waiters.fetch_sub(1);
      MutexError mutex_result = mutex->lock();
      if (!result.has_value() && result.error() == ETIMEDOUT)
        return CndVarResult::Timeout;
      return mutex_result == MutexError::NONE ? CndVarResult::Success
                                              : CndVarResult::MutexError;
    }

    CndWaiter waiter{};
    // Register the waiter to the queue.
    {
      cpp::lock_guard lock(queue_lock);
      waiter_queue.push_back(&waiter);
    }

    // Unlock the mutex and wait for the signal.
    mutex->unlock();
    bool locked = waiter.barrier.lock(timeout, /*is_shared=*/false);

    // if we wake up and find that we are still waiting, this means
    // timeout has been reached.
    uint8_t old_state = Waiting;
    if (waiter.state.compare_exchange_strong(old_state, Cancelled,
                                             cpp::MemoryOrder::ACQ_REL)) {
      // we haven't consumed the signal before timeout reaches.
      {
        cpp::lock_guard lock(queue_lock);
        WaiterHeader::remove(&waiter);
      }
      waiter.confirm_cancellation();
    } else if (!locked) {
      // Whenever a signal is already consumed, we compete for the mutex
      // in the FIFO order of the queue.
      waiter.barrier.lock();
    }

    MutexError mutex_result = mutex->lock();
    if (waiter.next != &waiter) {
      auto *next_waiter = static_cast<CndWaiter *>(waiter.next);
      WaiterHeader::remove(&waiter);
      auto &next_barrier_futex = next_waiter->barrier.get_raw_futex();
      auto &mutex_futex = mutex->get_raw_futex();
      FutexWordType prev = next_barrier_futex.exchange(
          RawMutex::UNLOCKED, cpp::MemoryOrder::RELEASE);
      if (prev == RawMutex::IN_CONTENTION)
        if (!mutex->can_be_requeued() ||
            !next_barrier_futex
                 .requeue_to(mutex_futex, cpp::nullopt, 0, 1,
                             /*is_shared=*/false)
                 .has_value())
          next_waiter->barrier.wake(/*is_shared=*/false);
    }
    if (mutex_result != MutexError::NONE)
      return CndVarResult::MutexError;
    return old_state == Waiting ? CndVarResult::Timeout : CndVarResult::Success;
  }

private:
  LIBC_INLINE void notify(bool broadcast, bool is_shared = false) {
    if (is_shared) {
      if (shared_waiters.load() == 0)
        return;
      shared_futex.fetch_add(1);
      if (broadcast)
        shared_futex.notify_all();
      else
        shared_futex.notify_one();
      return;
    }

    Futex sender_futex{0};
    auto wait_unregisteration_finish = [&]() {
      constexpr size_t LIMIT = 100;
      for (size_t i = 0; i < LIMIT; ++i) {
        if (sender_futex.load(cpp::MemoryOrder::RELAXED) == 0)
          break;
        sleep_briefly();
      }
      while (auto remaining = sender_futex.load(cpp::MemoryOrder::RELAXED))
        sender_futex.wait(remaining, cpp::nullopt, /*is_pshared=*/false);
    };
    CndWaiter *head = nullptr;
    CndWaiter *cursor = nullptr;
    size_t limit = broadcast ? cpp::numeric_limits<size_t>::max() : 1;
    {
      cpp::lock_guard lock(queue_lock);
      for (cursor = static_cast<CndWaiter *>(waiter_queue.next);
           cursor != &waiter_queue;
           cursor = static_cast<CndWaiter *>(cursor->next)) {
        if (limit == 0)
          break;
        uint8_t expected = Waiting;
        if (!cursor->state.compare_exchange_strong(expected, Signalled)) {
          sender_futex.fetch_add(1);
          cursor->sender_futex.store(&sender_futex);
          continue;
        }
        if (!head)
          head = cursor;
        limit--;
      }
      // remove everything before cursor
      auto removed_head = waiter_queue.next;
      auto removed_tail = cursor->prev;
      waiter_queue.next = cursor;
      cursor->prev = &waiter_queue;
      removed_tail->next = removed_head;
      removed_head->prev = removed_tail;
    }
    wait_unregisteration_finish();
    if (head)
      head->barrier.unlock();
  }

public:
  LIBC_INLINE void notify_one(bool is_shared = false) { notify(1, is_shared); }

  LIBC_INLINE void broadcast(bool is_shared = false) {
    notify(cpp::numeric_limits<size_t>::max(), is_shared);
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_CNDVAR_H
