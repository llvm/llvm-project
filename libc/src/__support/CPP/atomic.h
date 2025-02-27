//===-- A simple equivalent of std::atomic ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_CPP_ATOMIC_H
#define LLVM_LIBC_SRC___SUPPORT_CPP_ATOMIC_H

#include "src/__support/CPP/type_traits/has_unique_object_representations.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

#include "type_traits.h"

namespace LIBC_NAMESPACE_DECL {
namespace cpp {

enum class MemoryOrder : int {
  RELAXED = __ATOMIC_RELAXED,
  CONSUME = __ATOMIC_CONSUME,
  ACQUIRE = __ATOMIC_ACQUIRE,
  RELEASE = __ATOMIC_RELEASE,
  ACQ_REL = __ATOMIC_ACQ_REL,
  SEQ_CST = __ATOMIC_SEQ_CST
};

// These are a clang extension, see the clang documenation for more information:
// https://clang.llvm.org/docs/LanguageExtensions.html#scoped-atomic-builtins.
enum class MemoryScope : int {
#if defined(__MEMORY_SCOPE_SYSTEM) && defined(__MEMORY_SCOPE_DEVICE)
  SYSTEM = __MEMORY_SCOPE_SYSTEM,
  DEVICE = __MEMORY_SCOPE_DEVICE,
#else
  SYSTEM = 0,
  DEVICE = 0,
#endif
};

template <typename T> struct Atomic {
  static_assert(is_trivially_copyable_v<T> && is_copy_constructible_v<T> &&
                    is_move_constructible_v<T> && is_copy_assignable_v<T> &&
                    is_move_assignable_v<T>,
                "atomic<T> requires T to be trivially copyable, copy "
                "constructible, move constructible, copy assignable, "
                "and move assignable.");

  static_assert(cpp::has_unique_object_representations_v<T>,
                "atomic<T> in libc only support types whose values has unique "
                "object representations.");

private:
  // type conversion helper to avoid long c++ style casts
  LIBC_INLINE static int order(MemoryOrder mem_ord) {
    return static_cast<int>(mem_ord);
  }

  LIBC_INLINE static int scope(MemoryScope mem_scope) {
    return static_cast<int>(mem_scope);
  }

  LIBC_INLINE static T *addressof(T &ref) { return __builtin_addressof(ref); }

  // Require types that are 1, 2, 4, 8, or 16 bytes in length to be aligned to
  // at least their size to be potentially used lock-free.
  LIBC_INLINE_VAR static constexpr size_t MIN_ALIGNMENT =
      (sizeof(T) & (sizeof(T) - 1)) || (sizeof(T) > 16) ? 0 : sizeof(T);

  LIBC_INLINE_VAR static constexpr size_t ALIGNMENT = alignof(T) > MIN_ALIGNMENT
                                                          ? alignof(T)
                                                          : MIN_ALIGNMENT;

public:
  using value_type = T;

  // We keep the internal value public so that it can be addressable.
  // This is useful in places like the Linux futex operations where
  // we need pointers to the memory of the atomic values. Load and store
  // operations should be performed using the atomic methods however.
  alignas(ALIGNMENT) value_type val;

  LIBC_INLINE constexpr Atomic() = default;

  // Intializes the value without using atomic operations.
  LIBC_INLINE constexpr Atomic(value_type v) : val(v) {}

  LIBC_INLINE Atomic(const Atomic &) = delete;
  LIBC_INLINE Atomic &operator=(const Atomic &) = delete;

  // Atomic load.
  LIBC_INLINE operator T() { return load(); }

  LIBC_INLINE T
  load(MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
       [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    T res;
#if __has_builtin(__scoped_atomic_load)
    __scoped_atomic_load(addressof(val), addressof(res), order(mem_ord),
                         scope(mem_scope));
#else
    __atomic_load(addressof(val), addressof(res), order(mem_ord));
#endif
    return res;
  }

  // Atomic store.
  LIBC_INLINE T operator=(T rhs) {
    store(rhs);
    return rhs;
  }

  LIBC_INLINE void
  store(T rhs, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
        [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
#if __has_builtin(__scoped_atomic_store)
    __scoped_atomic_store(addressof(val), addressof(rhs), order(mem_ord),
                          scope(mem_scope));
#else
    __atomic_store(addressof(val), addressof(rhs), order(mem_ord));
#endif
  }

  // Atomic compare exchange
  LIBC_INLINE bool compare_exchange_strong(
      T &expected, T desired, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
      [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    return __atomic_compare_exchange(addressof(val), addressof(expected),
                                     addressof(desired), false, order(mem_ord),
                                     order(mem_ord));
  }

  // Atomic compare exchange (separate success and failure memory orders)
  LIBC_INLINE bool compare_exchange_strong(
      T &expected, T desired, MemoryOrder success_order,
      MemoryOrder failure_order,
      [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    return __atomic_compare_exchange(
        addressof(val), addressof(expected), addressof(desired), false,
        order(success_order), order(failure_order));
  }

  // Atomic compare exchange (weak version)
  LIBC_INLINE bool compare_exchange_weak(
      T &expected, T desired, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
      [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    return __atomic_compare_exchange(addressof(val), addressof(expected),
                                     addressof(desired), true, order(mem_ord),
                                     order(mem_ord));
  }

  // Atomic compare exchange (weak version with separate success and failure
  // memory orders)
  LIBC_INLINE bool compare_exchange_weak(
      T &expected, T desired, MemoryOrder success_order,
      MemoryOrder failure_order,
      [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    return __atomic_compare_exchange(
        addressof(val), addressof(expected), addressof(desired), true,
        order(success_order), order(failure_order));
  }

  LIBC_INLINE T
  exchange(T desired, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
           [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    T ret;
#if __has_builtin(__scoped_atomic_exchange)
    __scoped_atomic_exchange(addressof(val), addressof(desired), addressof(ret),
                             order(mem_ord), scope(mem_scope));
#else
    __atomic_exchange(addressof(val), addressof(desired), addressof(ret),
                      order(mem_ord));
#endif
    return ret;
  }

  LIBC_INLINE T
  fetch_add(T increment, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
            [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    static_assert(cpp::is_integral_v<T>, "T must be an integral type.");
#if __has_builtin(__scoped_atomic_fetch_add)
    return __scoped_atomic_fetch_add(addressof(val), increment, order(mem_ord),
                                     scope(mem_scope));
#else
    return __atomic_fetch_add(addressof(val), increment, order(mem_ord));
#endif
  }

  LIBC_INLINE T
  fetch_or(T mask, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
           [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    static_assert(cpp::is_integral_v<T>, "T must be an integral type.");
#if __has_builtin(__scoped_atomic_fetch_or)
    return __scoped_atomic_fetch_or(addressof(val), mask, order(mem_ord),
                                    scope(mem_scope));
#else
    return __atomic_fetch_or(addressof(val), mask, order(mem_ord));
#endif
  }

  LIBC_INLINE T
  fetch_and(T mask, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
            [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    static_assert(cpp::is_integral_v<T>, "T must be an integral type.");
#if __has_builtin(__scoped_atomic_fetch_and)
    return __scoped_atomic_fetch_and(addressof(val), mask, order(mem_ord),
                                     scope(mem_scope));
#else
    return __atomic_fetch_and(addressof(val), mask, order(mem_ord));
#endif
  }

  LIBC_INLINE T
  fetch_sub(T decrement, MemoryOrder mem_ord = MemoryOrder::SEQ_CST,
            [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
    static_assert(cpp::is_integral_v<T>, "T must be an integral type.");
#if __has_builtin(__scoped_atomic_fetch_sub)
    return __scoped_atomic_fetch_sub(addressof(val), decrement, order(mem_ord),
                                     scope(mem_scope));
#else
    return __atomic_fetch_sub(addressof(val), decrement, order(mem_ord));
#endif
  }

  // Set the value without using an atomic operation. This is useful
  // in initializing atomic values without a constructor.
  LIBC_INLINE void set(T rhs) { val = rhs; }
};

// Issue a thread fence with the given memory ordering.
LIBC_INLINE void atomic_thread_fence(
    MemoryOrder mem_ord,
    [[maybe_unused]] MemoryScope mem_scope = MemoryScope::DEVICE) {
#if __has_builtin(__scoped_atomic_thread_fence)
  __scoped_atomic_thread_fence(static_cast<int>(mem_ord),
                               static_cast<int>(mem_scope));
#else
  __atomic_thread_fence(static_cast<int>(mem_ord));
#endif
}

// Establishes memory synchronization ordering of non-atomic and relaxed atomic
// accesses, as instructed by order, between a thread and a signal handler
// executed on the same thread. This is equivalent to atomic_thread_fence,
// except no instructions for memory ordering are issued. Only reordering of
// the instructions by the compiler is suppressed as order instructs.
LIBC_INLINE void atomic_signal_fence([[maybe_unused]] MemoryOrder mem_ord) {
#if __has_builtin(__atomic_signal_fence)
  __atomic_signal_fence(static_cast<int>(mem_ord));
#else
  // if the builtin is not ready, use asm as a full compiler barrier.
  asm volatile("" ::: "memory");
#endif
}

} // namespace cpp
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_CPP_ATOMIC_H
