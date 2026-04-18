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
#include "src/__support/CPP/new.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/futex_utils.h" // Futex
#include "src/__support/threads/mutex.h"       // Mutex
#include "src/__support/threads/raw_mutex.h"   // RawMutex

#ifndef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#define LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY 1
#endif

namespace LIBC_NAMESPACE_DECL {

enum class CndVarResult {
  Success,
  MutexError,
  Timeout,
};

class PrivateCndVar {
  class CancellationBarrier {
    LIBC_INLINE_VAR static constexpr size_t SPIN_LIMIT = 100;
    LIBC_INLINE_VAR static constexpr size_t CANCEL_STEP = 2;
    LIBC_INLINE_VAR static constexpr size_t SLEEPING_BIT = 1;

    // LSB indicates whether the waiter is in sleeping state.
    Futex futex;

  public:
    LIBC_INLINE CancellationBarrier() : futex(0) {}
    // Add one more notification request.
    LIBC_INLINE void add_one() {
      futex.fetch_add(CANCEL_STEP, cpp::MemoryOrder::RELAXED);
    }
    // Send notification to one waiter.
    LIBC_INLINE void notify() {
      FutexWordType res = futex.fetch_sub(CANCEL_STEP);
      // Only need to goto syscall if waiter is sleep and we are the last one
      if (res <= (CANCEL_STEP | SLEEPING_BIT) && (res & SLEEPING_BIT) != 0)
        futex.notify_one();
    }
    LIBC_INLINE void wait() {
      size_t spin = 0;
      while (auto remaining = futex.load(cpp::MemoryOrder::RELAXED)) {
        // Set LSB to 1 to indicate that the waiter is entering sleeping
        // state.
        FutexWordType new_val = remaining | SLEEPING_BIT;
        if (spin > SPIN_LIMIT &&
            futex.compare_exchange_strong(remaining, new_val)) {
          futex.wait(new_val, /*timeout=*/cpp::nullopt, /*is_pshared=*/false);
          futex.fetch_sub(1);
          spin = 0;
        }
        sleep_briefly();
        spin++;
      }
    }
  };

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
      if (LIBC_UNLIKELY(prev == nullptr))
        prev = next = this;
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
    cpp::Atomic<CancellationBarrier *> sender_futex;
    RawMutex barrier;
    cpp::Atomic<uint8_t> state;

    LIBC_INLINE CndWaiter()
        : WaiterHeader{}, sender_futex(nullptr), barrier{}, state{Waiting} {
      // this lock should always success as no contention is possible
      (void)barrier.try_lock();
    }

    LIBC_INLINE void confirm_cancellation() {
      if (CancellationBarrier *sender = sender_futex.load())
        sender->notify();
    }
  };

  /*
  Layout:

  struct {
    void * __wait_queue_prev;
    void * __wait_queue_next;
    __futex_word __futex;
  }
  */
  WaiterHeader waiter_queue;
  RawMutex queue_lock;

public:
  using Timeout = internal::AbsTimeout;

  LIBC_INLINE constexpr PrivateCndVar() : waiter_queue{}, queue_lock{} {}

  LIBC_INLINE void reset() {
    queue_lock.reset();
    waiter_queue.prev = nullptr;
    waiter_queue.next = nullptr;
  }

  // TODO: register callback for pthread cancellation
  LIBC_INLINE CndVarResult wait(Mutex *mutex,
                                cpp::optional<Timeout> timeout = cpp::nullopt) {
#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
    if (timeout)
      ensure_monotonicity(*timeout);
#endif

    CndWaiter waiter{};
    // Register the waiter to the queue.
    {
      cpp::lock_guard lock(queue_lock);
      waiter_queue.push_back(&waiter);
    }

    // Unlock the mutex and wait for the signal.
    mutex->unlock();
    // Notice that lock is already initialized as LOCKED. We abuse the LOCKED
    // state to indicate that the waiter is pending.
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
      // in the FIFO order of the queue. We only relock if we previously
      // wake up due to timeout. Otherwise, it means that our turn has
      // come, so we don't need to relock.
      waiter.barrier.lock();
    }

    MutexError mutex_result = mutex->lock();
    // We need to establish contention after lock, otherwise
    // requeued thread may clear the contention bit even though
    // there are still waiters behind it.
    mutex->get_raw_futex().store(RawMutex::IN_CONTENTION);
    // If there is other in the queue after us, we need to wake the next waiter.
    // If we cancelled, we should naturally have waiter.next == &waiter
    if (waiter.next != &waiter) {
      auto *next_waiter = static_cast<CndWaiter *>(waiter.next);
      WaiterHeader::remove(&waiter);
      auto &next_barrier_futex = next_waiter->barrier.get_raw_futex();
      auto &mutex_futex = mutex->get_raw_futex();
      // the following is basically an inlined version of mutex::unlock
      // but with requeue instead of wake if it is possible.
      FutexWordType prev = next_barrier_futex.exchange(
          RawMutex::UNLOCKED, cpp::MemoryOrder::RELEASE);
      // If next waiter in queue sleeps, it will establish contention its own
      // barrier
      if (prev == RawMutex::IN_CONTENTION) {
        if (mutex->can_be_requeued()) {
          ErrorOr<int> res = next_barrier_futex.requeue_to(
              mutex_futex, cpp::nullopt, /*wake_limit=*/0,
              /*requeue_limit=*/1,
              /*is_shared=*/false);
          if (!res.has_value()) // cannot requeue on this system
            next_waiter->barrier.wake(/*is_shared=*/false);
        } else { // cannot requeue under special lock mode
          next_waiter->barrier.wake(/*is_shared=*/false);
        }
      }
    }
    if (mutex_result != MutexError::NONE)
      return CndVarResult::MutexError;
    return old_state == Waiting ? CndVarResult::Timeout : CndVarResult::Success;
  }

private:
  LIBC_INLINE void notify(size_t limit) {
    CancellationBarrier cancellation_barrier{};
    CndWaiter *head = nullptr;
    CndWaiter *cursor = nullptr;
    {
      cpp::lock_guard lock(queue_lock);
      waiter_queue.ensure_queue_initialization();
      if (waiter_queue.next == &waiter_queue)
        return;
      for (cursor = static_cast<CndWaiter *>(waiter_queue.next);
           cursor != &waiter_queue;
           cursor = static_cast<CndWaiter *>(cursor->next)) {
        if (limit == 0)
          break;
        uint8_t expected = Waiting;
        if (!cursor->state.compare_exchange_strong(expected, Signalled)) {
          cancellation_barrier.add_one();
          cursor->sender_futex.store(&cancellation_barrier);
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
    cancellation_barrier.wait();
    if (head)
      head->barrier.unlock();
  }

public:
  LIBC_INLINE void notify_one() { notify(1); }
  LIBC_INLINE void broadcast() { notify(cpp::numeric_limits<size_t>::max()); }
};

class SharedCndVar {
  /*
  Layout:
    struct {
      cpp::Atomic<size_t> shared_waiters;
      Futex shared_futex;
    };
  */
  cpp::Atomic<size_t> shared_waiters;
  Futex shared_futex;

public:
  using Timeout = internal::AbsTimeout;

  LIBC_INLINE constexpr SharedCndVar() : shared_waiters(0), shared_futex{0} {}

  LIBC_INLINE void reset() {
    shared_waiters.store(0);
    shared_futex.store(0);
  }

  // TODO: register callback for pthread cancellation
  LIBC_INLINE CndVarResult wait(Mutex *mutex,
                                cpp::optional<Timeout> timeout = cpp::nullopt) {
#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
    if (timeout)
      ensure_monotonicity(*timeout);
#endif

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

private:
  LIBC_INLINE void notify(bool broadcast) {
    if (shared_waiters.load() == 0)
      return;
    shared_futex.fetch_add(1);
    if (broadcast)
      shared_futex.notify_all();
    else
      shared_futex.notify_one();
    return;
  }

public:
  LIBC_INLINE void notify_one() { notify(/*broadcast=*/false); }
  LIBC_INLINE void broadcast() { notify(/*broadcast=*/true); }
};

class CndVar {
  union {
    PrivateCndVar private_cnd_var{};
    SharedCndVar shared_cnd_var;
  } storage;
  bool is_shared;

public:
  using Timeout = internal::AbsTimeout;
  LIBC_INLINE constexpr CndVar(bool is_shared) : is_shared(is_shared) {
    if (is_shared)
      new (&storage.shared_cnd_var) SharedCndVar();
    else
      new (&storage.private_cnd_var) PrivateCndVar();
  }
  LIBC_INLINE void reset() {
    if (is_shared)
      storage.shared_cnd_var.reset();
    else
      storage.private_cnd_var.reset();
  }
  LIBC_INLINE CndVarResult wait(Mutex *mutex,
                                cpp::optional<Timeout> timeout = cpp::nullopt) {
    if (is_shared)
      return storage.shared_cnd_var.wait(mutex, timeout);
    else
      return storage.private_cnd_var.wait(mutex, timeout);
  }
  LIBC_INLINE void notify_one() {
    if (is_shared)
      storage.shared_cnd_var.notify_one();
    else
      storage.private_cnd_var.notify_one();
  }
  LIBC_INLINE void broadcast() {
    if (is_shared)
      storage.shared_cnd_var.broadcast();
    else
      storage.private_cnd_var.broadcast();
  }
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_THREADS_CNDVAR_H
