//===--- Implementation of a Linux RwLock class ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_RWLOCK_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_RWLOCK_H

#include "hdr/errno_macros.h"
#include "hdr/types/pid_t.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/optional.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/optimization.h"
#include "src/__support/threads/linux/futex_utils.h"
#include "src/__support/threads/linux/futex_word.h"
#include "src/__support/threads/linux/raw_mutex.h"
#include "src/__support/threads/sleep.h"

#ifndef LIBC_COPT_RWLOCK_DEFAULT_SPIN_COUNT
#define LIBC_COPT_RWLOCK_DEFAULT_SPIN_COUNT 100
#endif

#ifndef LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#define LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY 1
#warning "LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY is not defined, defaulting to 1"
#endif

#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
#include "src/__support/time/linux/monotonicity.h"
#endif

namespace LIBC_NAMESPACE {
class RwLock {
public:
  enum class Role : char { Reader, Writer };

private:
  class WaitingQueue final : private RawMutex {
    FutexWordType pending_reader;
    FutexWordType pending_writer;
    Futex reader_serialization;
    Futex writer_serialization;

  public:
    class Guard {
      WaitingQueue &queue;

      LIBC_INLINE constexpr Guard(WaitingQueue &queue) : queue(queue) {}

    public:
      LIBC_INLINE ~Guard() { queue.unlock(); }
      template <Role role> LIBC_INLINE FutexWordType &pending_count() {
        if constexpr (role == Role::Reader)
          return queue.pending_reader;
        else
          return queue.pending_writer;
      }
      template <Role role> LIBC_INLINE FutexWordType &serialization() {
        if constexpr (role == Role::Reader)
          return queue.reader_serialization.val;
        else
          return queue.writer_serialization.val;
      }
      friend WaitingQueue;
    };

  public:
    LIBC_INLINE constexpr WaitingQueue()
        : RawMutex(), pending_reader(0), pending_writer(0),
          reader_serialization(0), writer_serialization(0) {}
    LIBC_INLINE Guard acquire() {
      this->lock();
      return Guard(*this);
    }

    template <Role role>
    LIBC_INLINE long wait(FutexWordType expected,
                          cpp::optional<Futex::Timeout> timeout,
                          bool is_pshared) {
      if constexpr (role == Role::Reader)
        return reader_serialization.wait(expected, timeout, is_pshared);
      else
        return writer_serialization.wait(expected, timeout, is_pshared);
    }

    template <Role role> LIBC_INLINE long notify(bool is_pshared) {
      if constexpr (role == Role::Reader)
        return reader_serialization.notify_all(is_pshared);
      else
        return writer_serialization.notify_one(is_pshared);
    }
  };

public:
  enum class LockResult : int {
    Success = 0,
    TimedOut = ETIMEDOUT,
    Overflow = EAGAIN,
    Busy = EBUSY,
    Deadlock = EDEADLOCK,
    PermissionDenied = EPERM,
  };

private:
  // The State of the RwLock is stored in a 32-bit word, consisting of the
  // following components:
  // -----------------------------------------------
  // | Range |           Description               |
  // ===============================================
  // | 0     | Pending Reader Bit                  |
  // -----------------------------------------------
  // | 1     | Pending Writer Bit                  |
  // -----------------------------------------------
  // | 2-30  | Active Reader Count                 |
  // -----------------------------------------------
  // | 31    | Active Writer Bit                   |
  // -----------------------------------------------
  class State {
    // We use the signed interger as the state type. It is easier
    // to handle state trasitions and detections using signed integers.
    using Type = int32_t;

    // Shift amounts to access the components of the state.
    LIBC_INLINE_VAR static constexpr Type PENDING_READER_SHIFT = 0;
    LIBC_INLINE_VAR static constexpr Type PENDING_WRITER_SHIFT = 1;
    LIBC_INLINE_VAR static constexpr Type ACTIVE_READER_SHIFT = 2;
    LIBC_INLINE_VAR static constexpr Type ACTIVE_WRITER_SHIFT = 31;

    // Bitmasks to access the components of the state.
    LIBC_INLINE_VAR static constexpr Type PENDING_READER_BIT =
        1 << PENDING_READER_SHIFT;
    LIBC_INLINE_VAR static constexpr Type PENDING_WRITER_BIT =
        1 << PENDING_WRITER_SHIFT;
    LIBC_INLINE_VAR static constexpr Type ACTIVE_READER_COUNT_UNIT =
        1 << ACTIVE_READER_SHIFT;
    LIBC_INLINE_VAR static constexpr Type ACTIVE_WRITER_BIT =
        1 << ACTIVE_WRITER_SHIFT;
    LIBC_INLINE_VAR static constexpr Type PENDING_MASK =
        PENDING_READER_BIT | PENDING_WRITER_BIT;

  private:
    Type state;

  public:
    // Construction and conversion functions.
    LIBC_INLINE constexpr State(Type state = 0) : state(state) {}
    LIBC_INLINE constexpr operator Type() const { return state; }

    // Utilities to check the state of the RwLock.
    LIBC_INLINE constexpr bool has_active_writer() const { return state < 0; }
    LIBC_INLINE constexpr bool has_active_reader() const {
      return state > ACTIVE_READER_COUNT_UNIT;
    }
    LIBC_INLINE constexpr bool has_acitve_owner() const {
      return has_active_reader() || has_active_writer();
    }
    LIBC_INLINE constexpr bool has_last_reader() const {
      return (state >> ACTIVE_READER_SHIFT) == 1;
    }
    LIBC_INLINE constexpr bool has_pending_writer() const {
      return state & PENDING_WRITER_BIT;
    }
    LIBC_INLINE constexpr bool has_pending() const {
      return state & PENDING_MASK;
    }
    LIBC_INLINE constexpr State set_writer_bit() const {
      return State(state | ACTIVE_WRITER_BIT);
    }

    // The preference parameter changes the behavior of the lock acquisition
    // if there are both readers and writers waiting for the lock. If writers
    // are preferred, reader acquisition will be blocked until all pending
    // writers are served.
    template <Role role> LIBC_INLINE bool can_acquire(Role preference) const {
      if constexpr (role == Role::Reader) {
        switch (preference) {
        case Role::Reader:
          return !has_active_writer();
        case Role::Writer:
          return !has_active_writer() && !has_pending_writer();
        }
      } else
        return !has_acitve_owner();
    }

    // This function check if it is possible to grow the reader count without
    // overflowing the state.
    LIBC_INLINE cpp::optional<State> try_increase_reader_count() const {
      LIBC_ASSERT(!has_active_writer() &&
                  "try_increase_reader_count shall only be called when there "
                  "is no active writer.");
      State res;
      if (LIBC_UNLIKELY(__builtin_sadd_overflow(state, ACTIVE_READER_COUNT_UNIT,
                                                &res.state)))
        return cpp::nullopt;
      return res;
    }

    // Utilities to do atomic operations on the state.
    LIBC_INLINE static State
    fetch_sub_reader_count(cpp::Atomic<Type> &target,
                           cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      return State(target.fetch_sub(ACTIVE_READER_COUNT_UNIT, order));
    }

    LIBC_INLINE static State
    load(cpp::Atomic<Type> &target,
         cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      return State(target.load(order));
    }

    template <Role role>
    LIBC_INLINE static State
    fetch_set_pending_bit(cpp::Atomic<Type> &target,
                          cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      if constexpr (role == Role::Reader)
        return State(target.fetch_or(PENDING_READER_BIT, order));
      else
        return State(target.fetch_or(PENDING_WRITER_BIT, order));
    }
    template <Role role>
    LIBC_INLINE static State fetch_clear_pending_bit(
        cpp::Atomic<Type> &target,
        cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      if constexpr (role == Role::Reader)
        return State(target.fetch_and(~PENDING_READER_BIT, order));
      else
        return State(target.fetch_and(~PENDING_WRITER_BIT, order));
    }
    LIBC_INLINE static State fetch_set_active_writer(
        cpp::Atomic<Type> &target,
        cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      return State(target.fetch_or(ACTIVE_WRITER_BIT, order));
    }
    LIBC_INLINE static State fetch_clear_active_writer(
        cpp::Atomic<Type> &target,
        cpp::MemoryOrder order = cpp::MemoryOrder::SEQ_CST) {
      return State(target.fetch_and(~ACTIVE_WRITER_BIT, order));
    }

    LIBC_INLINE bool
    compare_exchange_weak_with(cpp::Atomic<Type> &target, State desired,
                               cpp::MemoryOrder success_order,
                               cpp::MemoryOrder failure_order) {
      return target.compare_exchange_weak(state, desired, success_order,
                                          failure_order);
    }

    // Utilities to spin and reload the state.
  private:
    template <class F>
    LIBC_INLINE static State spin_reload_until(cpp::Atomic<Type> &target,
                                               F &&func, unsigned spin_count) {
      for (;;) {
        auto state = State::load(target);
        if (func(state) || spin_count == 0)
          return state;
        sleep_briefly();
        spin_count--;
      }
    }

  public:
    template <Role role>
    LIBC_INLINE static State spin_reload(cpp::Atomic<Type> &target,
                                         Role preference, unsigned spin_count) {
      if constexpr (role == Role::Reader) {
        // Return the reader state if either the lock is available or there is
        // any
        // ongoing contention.
        return spin_reload_until(
            target,
            [=](State state) {
              return state.can_acquire<Role::Reader>(preference) ||
                     state.has_pending();
            },
            spin_count);
      } else {
        // Return the writer state if either the lock is available or there is
        // any
        // contention *between writers*. Since writers can be way less than
        // readers, we allow them to spin more to improve the fairness.
        return spin_reload_until(
            target,
            [=](State state) {
              return state.can_acquire<Role::Writer>(preference) ||
                     state.has_pending_writer();
            },
            spin_count);
      }
    }
  };

private:
  // Whether the RwLock is shared between processes.
  bool is_pshared;
  // Reader/Writer preference.
  Role preference;
  // State to keep track of the RwLock.
  cpp::Atomic<int32_t> state;
  // writer_tid is used to keep track of the thread id of the writer. Notice
  // that TLS address is not a good idea here since it may remains the same
  // across forked processes.
  cpp::Atomic<pid_t> writer_tid;
  // Waiting queue to keep track of the pending readers and writers.
  WaitingQueue queue;

private:
  // TODO: use cached thread id once implemented.
  LIBC_INLINE static pid_t gettid() { return syscall_impl<pid_t>(SYS_gettid); }

  template <Role role> LIBC_INLINE LockResult try_lock(State &old) {
    if constexpr (role == Role::Reader) {
      while (LIBC_LIKELY(old.can_acquire<Role::Reader>(preference))) {
        cpp::optional<State> next = old.try_increase_reader_count();
        if (!next)
          return LockResult::Overflow;
        if (LIBC_LIKELY(old.compare_exchange_weak_with(
                state, *next, cpp::MemoryOrder::ACQUIRE,
                cpp::MemoryOrder::RELAXED)))
          return LockResult::Success;
        // Notice that old is updated by the compare_exchange_weak_with
        // function.
      }
      return LockResult::Busy;
    } else {
      // This while loop should terminate quickly
      while (LIBC_LIKELY(old.can_acquire<Role::Writer>(preference))) {
        if (LIBC_LIKELY(old.compare_exchange_weak_with(
                state, old.set_writer_bit(), cpp::MemoryOrder::ACQUIRE,
                cpp::MemoryOrder::RELAXED))) {
          writer_tid.store(gettid(), cpp::MemoryOrder::RELAXED);
          return LockResult::Success;
        }
        // Notice that old is updated by the compare_exchange_weak_with
        // function.
      }
      return LockResult::Busy;
    }
  }

public:
  LIBC_INLINE constexpr RwLock(Role preference = Role::Reader,
                               bool is_pshared = false)
      : is_pshared(is_pshared), preference(preference), state(0), writer_tid(0),
        queue() {}

  LIBC_INLINE LockResult try_read_lock() {
    State old = State::load(state, cpp::MemoryOrder::RELAXED);
    return try_lock<Role::Reader>(old);
  }
  LIBC_INLINE LockResult try_write_lock() {
    State old = State::load(state, cpp::MemoryOrder::RELAXED);
    return try_lock<Role::Writer>(old);
  }

private:
  template <Role role>
  LIBC_INLINE LockResult
  lock_slow(cpp::optional<Futex::Timeout> timeout = cpp::nullopt,
            unsigned spin_count = LIBC_COPT_RWLOCK_DEFAULT_SPIN_COUNT) {
    // Phase 1: deadlock detection.
    // A deadlock happens if this is a RAW/WAW lock in the same thread.
    if (writer_tid.load(cpp::MemoryOrder::RELAXED) == gettid())
      return LockResult::Deadlock;

#if LIBC_COPT_TIMEOUT_ENSURE_MONOTONICITY
    // Phase 2: convert the timeout if necessary.
    if (timeout)
      ensure_monotonicity(*timeout);
#endif

    // Phase 3: spin to get the initial state. We ignore the timing due to
    // spin since it should end quickly.
    State old = State::spin_reload<role>(state, preference, spin_count);

    // Enter the main acquisition loop.
    for (;;) {
      // Phase 4: if the lock can be acquired, try to acquire it.
      LockResult result = try_lock<role>(old);
      if (result != LockResult::Busy)
        return result;

      // Phase 5: register ourselves as a pending reader.
      int serial_number;
      {
        // The queue need to be protected by a mutex since the operations in
        // this block must be executed as a whole transaction. It is possible
        // that this lock will make the timeout imprecise, but this is the
        // best we can do. The transaction is small and everyone should make
        // progress rather quickly.
        WaitingQueue::Guard guard = queue.acquire();
        guard.template pending_count<role>()++;

        // Use atomic operation to guarantee the total order of the operations
        // on the state. The pending flag update should be visible to any
        // succeeding unlock events. Or, if a unlock does happen before we
        // sleep on the futex, we can avoid such waiting.
        old = State::fetch_set_pending_bit<role>(state,
                                                 cpp::MemoryOrder::RELAXED);
        // no need to use atomic since it is already protected by the mutex.
        serial_number = guard.serialization<role>();
      }

      // Phase 6: do futex wait until the lock is available or timeout is
      // reached.
      bool timeout_flag = false;
      if (!old.can_acquire<role>(preference)) {
        timeout_flag = (queue.wait<role>(serial_number, timeout, is_pshared) ==
                        -ETIMEDOUT);

        // Phase 7: unregister ourselves as a pending reader.
        {
          // Similarly, the unregister operation should also be an atomic
          // transaction.
          WaitingQueue::Guard guard = queue.acquire();
          guard.pending_count<role>()--;
          // Clear the flag if we are the last reader. The flag must be
          // cleared otherwise operations like trylock may fail even though
          // there is no competitors.
          if (guard.pending_count<role>() == 0)
            State::fetch_clear_pending_bit<role>(state,
                                                 cpp::MemoryOrder::RELAXED);
        }

        // Phase 8: exit the loop is timeout is reached.
        if (timeout_flag)
          return LockResult::TimedOut;

        // Phase 9: reload the state and retry the acquisition.
        old = State::spin_reload<role>(state, preference, spin_count);
      }
    }
  }

public:
  LIBC_INLINE LockResult
  read_lock(cpp::optional<Futex::Timeout> timeout = cpp::nullopt,
            unsigned spin_count = LIBC_COPT_RWLOCK_DEFAULT_SPIN_COUNT) {
    LockResult result = try_read_lock();
    if (LIBC_LIKELY(result != LockResult::Busy))
      return result;
    return lock_slow<Role::Reader>(timeout, spin_count);
  }
  LIBC_INLINE LockResult
  write_lock(cpp::optional<Futex::Timeout> timeout = cpp::nullopt,
             unsigned spin_count = LIBC_COPT_RWLOCK_DEFAULT_SPIN_COUNT) {
    LockResult result = try_write_lock();
    if (LIBC_LIKELY(result != LockResult::Busy))
      return result;
    return lock_slow<Role::Writer>(timeout, spin_count);
  }

private:
  // Compiler somehow decides that this function may be inlined, which leads to
  // a larger unlock function that is infeasible to be inlined. Since
  // notifcation routine is colder we mark it as noinline explicitly.
  [[gnu::noinline]]
  LIBC_INLINE void notify_pending_threads() {
    enum class WakeTarget { Readers, Writers, None };
    WakeTarget status;

    {
      WaitingQueue::Guard guard = queue.acquire();
      if (guard.pending_count<Role::Writer>() != 0) {
        guard.serialization<Role::Writer>()++;
        status = WakeTarget::Writers;
      } else if (guard.pending_count<Role::Reader>() != 0) {
        guard.serialization<Role::Reader>()++;
        status = WakeTarget::Readers;
      } else
        status = WakeTarget::None;
    }

    if (status == WakeTarget::Readers)
      queue.notify<Role::Reader>(is_pshared);
    else if (status == WakeTarget::Writers)
      queue.notify<Role::Writer>(is_pshared);
  }

public:
  LIBC_INLINE LockResult unlock() {
    State old = State::load(state, cpp::MemoryOrder::RELAXED);
    if (old.has_active_writer()) {
      // The lock is held by a writer.
      // Check if we are the owner of the lock.
      if (writer_tid.load(cpp::MemoryOrder::RELAXED) != gettid())
        return LockResult::PermissionDenied;
      // clear writer tid.
      writer_tid.store(0, cpp::MemoryOrder::RELAXED);
      // clear the writer bit.
      old = State::fetch_clear_active_writer(state);
      // If there is no pending readers or writers, we are done.
      if (!old.has_pending())
        return LockResult::Success;
    } else if (old.has_active_reader()) {
      // The lock is held by readers.
      // Decrease the reader count.
      old = State::fetch_sub_reader_count(state);
      // If there is no pending readers or writers, we are done.
      if (!old.has_last_reader() || !old.has_pending())
        return LockResult::Success;
    } else
      return LockResult::PermissionDenied;

    notify_pending_threads();
    return LockResult::Success;
  }

  LIBC_INLINE LockResult check_for_destroy() {
    State old = State::load(state, cpp::MemoryOrder::RELAXED);
    if (old.has_acitve_owner())
      return LockResult::Busy;
    return LockResult::Success;
  }
};
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_RWLOCK_H
