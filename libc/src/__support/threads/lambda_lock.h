//===-- MCS-based Flat-Combining Lambda Lock --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_LAMBDA_LOCK_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_LAMBDA_LOCK_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/limits.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/threads/linux/futex_utils.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/__support/threads/sleep.h"
#include "src/__support/threads/spin_lock.h"

// This file contains an implementation of a flat combining lock based on MCS
// queue.
//
// What is an MCS queue?
// =====================
// An MCS queue is a queue-based lock that is designed to be cache-friendly.
// Each thread only spin on its local node, with minimal traffic across threads.
// The thread-local node is not nessarily a node inside TLS storage. Rather, the
// node can be allocated on stack. It is assumed that, during the lifespan of
// the node, the thread is waiting in its locking routine, thus the stack space
// is always valid.
//
// ┌──────┐        ┌────────────────────┐      ┌────────────────────┐
// │ Tail │        │  Thread 1 (Stack)  │      │  Thread 2 (Stack)  │
// └──┬───┘        ├────────────────────┤      ├────────────────────┤
//    │            │  ┌──────────────┐  │ next │                    │
//    └────────────┼─>│    Node N    │<─┼──┐   │     ..........     │
//                 │  └──────────────┘  │  │   │   ┌─────────────┐  │ next
//                 │     ..........     │  └───┼───┤  Node N-1   │<─┼─── ....
//                 │                    │      │   └─────────────┘  │
//                 └────────────────────┘      └────────────────────┘
//
// The queue is processed in a FIFO manner. When submitting a task to the lock
// queue, the thread swaps its own node with the global tail pointer to register
// itself to the queue. The thread then waits until its own turn. This is
// usually signaled by the previous node in the queue. Once the thread receives
// the signal, it executes the task and then signals the next node in the queue.
//
// What is a flat combining lock?
// ==============================
// Normally, a Mutex maintains the exclusivity using a lock word. Threads
// polls/parks on that lock word until it finds an opportunity to acquire the
// lock by a successful posting to the lock word. Then the thread go ahead to do
// its task on the shared data. In heavy contention, however, the shared data is
// bouncing among threads, causing a lot of extra traffic.
// Flat combining is a technique to resolve such problem. The general idea is
// that when a thread acquires the lock and finished its critical section, its
// cache has the "ownership" of the shared data. Instead of passing such
// ownership to the next thread, the current thread can continue to execute the
// the crtitical section on behalf of the next thread and signal the next thread
// once the task is done.
//
// How to achieve flat combining with MCS queue?
// =============================================
// We can use the queue itself specifies the job waiting to be done by adding
// additional fields to the node such as a function pointer to the critical
// section. When a thread acquires the lock on the head of the queue, it begins
// to follow the pointers among the nodes and execute their critical sections.
// The exclusivity is garanteed by the fact that the combiner thread always
// holds a global lock. Such lock is passed to the next combiner if needed.
// Instead of let each individual thread to propagate the finishing signal, the
// combiner just notify each waiting thread onces their critical section is
// executed.
//
// Pros and Cons
// =============
// Pros:
// - Flat-combing to reduce coherence traffic.
// - Threads are served in FIFO order.
// Cons:
// - Access to thread-local data needs to be carefully managed (e.g. by passing
// pointers to the desired TLS slot if needed).
// - The combiner thread may execute mutiple critical sections in a row. One may
// want limit the combination sometimes.
// - Stack corruption may corrupt the whole waiting queue. Thread cancellation
// should not be allowed during flat combining.
//
// Why does it matter?
// ===================
// Its pros certainly matter. Flat-combining is new synchronization paradigm.
// Many systems actually want to introduce such mechanism but hindered by
// existing codebase and many works are done to progressive migration. [1] shows
// that flat combining can significantly improve the performance inside Linux
// kernel. SnMalloc recently introduced flat combining to speed up its
// initialization process [2]. As llvm-libc is still relatively new, we have
// opportunities to embrace flat-combining from the beginning rather migrating
// to it later on.
//
// [1]: Ship your Critical Section, Not Your Data: Enabling Transparent
// Delegation with TCLOCKS (OSDI '23)
// https://www.usenix.org/conference/osdi23/presentation/gupta)
// [2]: SnMalloc: Implement MCS Combining lock
// https://github.com/microsoft/snmalloc/commit/6af38acd94d401313b7237bb7c2df5e662e135cf
namespace LIBC_NAMESPACE_DECL {
// On stack lock node header for MCS-based flat combining lock.
class RawLambdaLockHeader {
  // Status of the futex.
  Futex status;
  // Next pointer.
  cpp::Atomic<RawLambdaLockHeader *> next;
  // Lambda function to execute.
  void (*lambda)(RawLambdaLockHeader &);

  // Possible values for the status:
  // - WAITING: the task asscociated with the node is not finished.
  // - DONE: the task associated with the node is finished.
  // - COMBINING: the task is not finished and target thread is the combiner.
  // - SLEEPING: the task is not finished and target thread is sleeping.
  LIBC_INLINE_VAR static constexpr FutexWordType WAITING = 0;
  LIBC_INLINE_VAR static constexpr FutexWordType DONE = 1;
  LIBC_INLINE_VAR static constexpr FutexWordType COMBINING = 2;
  LIBC_INLINE_VAR static constexpr FutexWordType SLEEPING = 3;

  friend class RawLambdaLock;

public:
  LIBC_INLINE constexpr RawLambdaLockHeader(
      void (*lambda)(RawLambdaLockHeader &))
      : status(WAITING), next(nullptr), lambda(lambda) {}
};

// Helper to maintain the lock ownership.
class RawLambdaLockCombinerToken {
  bool handed_over;
  SpinLock &lock;

  LIBC_INLINE RawLambdaLockCombinerToken(SpinLock &lock)
      : handed_over(false), lock(lock) {}

public:
  // Notice that the lock ownership is "announced" rather than "acquired".
  LIBC_INLINE static RawLambdaLockCombinerToken announce(SpinLock &lock) {
    LIBC_ASSERT(lock.is_locked());
    return {lock};
  }

  LIBC_INLINE
  RawLambdaLockCombinerToken(const RawLambdaLockCombinerToken &) = delete;

  LIBC_INLINE
  RawLambdaLockCombinerToken(RawLambdaLockCombinerToken &&that)
      : handed_over(that.handed_over), lock(that.lock) {
    that.handed_over = true;
  };

  // Hand over the lock ownership.
  LIBC_INLINE void hand_over() { handed_over = true; }

  LIBC_INLINE ~RawLambdaLockCombinerToken() {
    LIBC_ASSERT(handed_over || lock.is_locked());
    if (!handed_over)
      lock.unlock();
  }
};

// Global status of the lock.
class RawLambdaLock {
  LIBC_INLINE_VAR static constexpr int SPIN_COUNT = 100;
  // tail of the waiting queue.
  cpp::Atomic<RawLambdaLockHeader *> tail;
  // spin lock to protect the queue.
  SpinLock spin_lock;

public:
  LIBC_INLINE constexpr RawLambdaLock() : tail(nullptr), spin_lock() {}

  LIBC_INLINE cpp::optional<RawLambdaLockCombinerToken>
  try_lock_without_queueing() {
    if (tail.load(cpp::MemoryOrder::RELAXED) == nullptr && spin_lock.try_lock())
      return RawLambdaLockCombinerToken::announce(spin_lock);
    return cpp::nullopt;
  }

  LIBC_INLINE void enqueue(RawLambdaLockHeader &header,
                           size_t combining_limit) {
    // Set ourself as the tail and acquire previous tail.
    RawLambdaLockHeader *prev =
        tail.exchange(&header, cpp::MemoryOrder::ACQ_REL);

    if (prev) {
      // If there is a previous waiter, we should stay put until hearing back
      // from the previous node. To do so, we need to register ourself to the
      // next pointer of the previous node.
      prev->next.store(&header, cpp::MemoryOrder::RELEASE);
      int remaining_spins = SPIN_COUNT;
      // Do spin polling for certain amount of time.
      while (remaining_spins > 0) {
        if (header.status.load(cpp::MemoryOrder::RELAXED) !=
            RawLambdaLockHeader::WAITING)
          break;
        sleep_briefly();
        remaining_spins--;
      }
      // If we used up all spins, we may need to go to sleep.
      if (remaining_spins == 0) {
        FutexWordType expected = RawLambdaLockHeader::WAITING;
        if (header.status.compare_exchange_strong(
                expected, RawLambdaLockHeader::SLEEPING,
                cpp::MemoryOrder::ACQ_REL, cpp::MemoryOrder::RELAXED))
          header.status.wait(RawLambdaLockHeader::SLEEPING);
      }
      // We are here if and only if the status is modified by previous waiter.
      FutexWordType status = header.status.load(cpp::MemoryOrder::ACQUIRE);
      if (status == RawLambdaLockHeader::DONE)
        return;
      LIBC_ASSERT(status == RawLambdaLockHeader::COMBINING &&
                  spin_lock.is_locked());
      // We are the combiner: the lock ownership is handed over to us.
    } else
      // We are the combiner: but we need to acquire the lock on our own.
      spin_lock.lock();

    // Announce the lock ownership.
    RawLambdaLockCombinerToken token =
        RawLambdaLockCombinerToken::announce(spin_lock);

    RawLambdaLockHeader *current = &header;
    auto wakeup = [](RawLambdaLockHeader *node, FutexWordType new_status) {
      if (node->status.exchange(new_status, cpp::MemoryOrder::ACQ_REL) ==
          RawLambdaLockHeader::SLEEPING)
        node->status.notify_one();
    };
    size_t remaining = combining_limit;
    while (true) {
      // Execute the lambda function.
      current->lambda(*current);
      RawLambdaLockHeader *next = current->next.load(cpp::MemoryOrder::ACQUIRE);

      if (remaining > 0 && next != nullptr) {
        // Wake up the current node if it is sleeping.
        wakeup(current, RawLambdaLockHeader::DONE);
        current = next;
        remaining--;
        continue;
      }

      // We are here either because the combining limit is reached or the last
      // attempt to get the next node failed. Invariant: current node is
      // executed.

      // Try close current waiting queue.
      RawLambdaLockHeader *expected = current;
      if (tail.compare_exchange_strong(expected, nullptr,
                                       cpp::MemoryOrder::ACQ_REL,
                                       cpp::MemoryOrder::RELAXED)) {
        // The queue is closed. We are good to go, before which we need to
        // wake up the last node if it is sleeping.
        wakeup(current, RawLambdaLockHeader::DONE);
        return;
      }

      // We failed to close the queue, handover the combiner role to the next
      // one in queue.
      token.hand_over();
      while (!current->next.load(cpp::MemoryOrder::RELAXED))
        sleep_briefly();
      RawLambdaLockHeader *combiner =
          current->next.load(cpp::MemoryOrder::ACQUIRE);
      wakeup(combiner, RawLambdaLockHeader::COMBINING);
      wakeup(current, RawLambdaLockHeader::DONE);
      return;
    }
  }
};

template <typename T> class LambdaLock {
  RawLambdaLock inner;
  T data;

  template <typename F>
  [[gnu::cold]] LIBC_INLINE void enqueue_slow(F &&lambda,
                                              size_t combining_limit) {
    // Create the waiting node on stack and enqueue it.
    struct Node {
      RawLambdaLockHeader header;
      F lambda;
      T &data;
    };
    auto closure = [](RawLambdaLockHeader &header) {
      Node *node = reinterpret_cast<Node *>(&header);
      node->lambda(node->data);
    };
    Node node{{closure}, cpp::forward<F>(lambda), data};
    inner.enqueue(node.header, combining_limit);
  }

public:
  template <typename... U>
  LIBC_INLINE constexpr LambdaLock(U &&...args)
      : inner(), data(cpp::forward<U>(args)...) {}
  template <typename F>
  LIBC_INLINE void
  enqueue(F &&lambda,
          size_t combining_limit = cpp::numeric_limits<size_t>::max()) {
    // First try to lock without queueing. In this situation, we do not even
    // register ourself to the queue.
    // The futex sleeping happens only when prev node exists so the local
    // execution of the lambda function in this case won't put all other threads
    // to sleep. Hence, if contention really happens, there will be a failed one
    // to go into the slow path and become the real combiner.
    if (cpp::optional<RawLambdaLockCombinerToken> token =
            inner.try_lock_without_queueing()) {
      lambda(data);
      return;
    }
    enqueue_slow(cpp::forward<F>(lambda), combining_limit);
  }
  // This Should only be invoked in single-threaded context or other known
  // race-free context.
  T &get_unsafe() { return data; }
};
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LAMBDA_LOCK_H
