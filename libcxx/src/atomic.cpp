//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__thread/timed_backoff_policy.h>
#include <atomic>
#include <climits>
#include <functional>
#include <thread>

#include "include/apple_availability.h"

#ifdef __linux__

#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>

// libc++ uses SYS_futex as a universal syscall name. However, on 32 bit architectures
// with a 64 bit time_t, we need to specify SYS_futex_time64.
#  if !defined(SYS_futex) && defined(SYS_futex_time64)
#    define SYS_futex SYS_futex_time64
#  endif

#elif defined(__FreeBSD__)

#  include <sys/types.h>
#  include <sys/umtx.h>

#else // <- Add other operating systems here

// Baseline needs no new headers

#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef __linux__

static void
__libcpp_platform_wait_on_address(__cxx_atomic_contention_t const volatile* __ptr, __cxx_contention_t __val) {
  static constexpr timespec __timeout = {2, 0};
  syscall(SYS_futex, __ptr, FUTEX_WAIT_PRIVATE, __val, &__timeout, 0, 0);
}

static void __libcpp_platform_wake_by_address(__cxx_atomic_contention_t const volatile* __ptr, bool __notify_one) {
  syscall(SYS_futex, __ptr, FUTEX_WAKE_PRIVATE, __notify_one ? 1 : INT_MAX, 0, 0, 0);
}

#elif defined(__APPLE__) && defined(_LIBCPP_USE_ULOCK)

extern "C" int __ulock_wait(
    uint32_t operation, void* addr, uint64_t value, uint32_t timeout); /* timeout is specified in microseconds */
extern "C" int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value);

#  define UL_COMPARE_AND_WAIT 1
#  define ULF_WAKE_ALL 0x00000100

static void
__libcpp_platform_wait_on_address(__cxx_atomic_contention_t const volatile* __ptr, __cxx_contention_t __val) {
  __ulock_wait(UL_COMPARE_AND_WAIT, const_cast<__cxx_atomic_contention_t*>(__ptr), __val, 0);
}

static void __libcpp_platform_wake_by_address(__cxx_atomic_contention_t const volatile* __ptr, bool __notify_one) {
  __ulock_wake(
      UL_COMPARE_AND_WAIT | (__notify_one ? 0 : ULF_WAKE_ALL), const_cast<__cxx_atomic_contention_t*>(__ptr), 0);
}

#elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8
/*
 * Since __cxx_contention_t is int64_t even on 32bit FreeBSD
 * platforms, we have to use umtx ops that work on the long type, and
 * limit its use to architectures where long and int64_t are synonyms.
 */

static void
__libcpp_platform_wait_on_address(__cxx_atomic_contention_t const volatile* __ptr, __cxx_contention_t __val) {
  _umtx_op(const_cast<__cxx_atomic_contention_t*>(__ptr), UMTX_OP_WAIT, __val, NULL, NULL);
}

static void __libcpp_platform_wake_by_address(__cxx_atomic_contention_t const volatile* __ptr, bool __notify_one) {
  _umtx_op(const_cast<__cxx_atomic_contention_t*>(__ptr), UMTX_OP_WAKE, __notify_one ? 1 : INT_MAX, NULL, NULL);
}

#else // <- Add other operating systems here

// Baseline is just a timed backoff

static void
__libcpp_platform_wait_on_address(__cxx_atomic_contention_t const volatile* __ptr, __cxx_contention_t __val) {
  __libcpp_thread_poll_with_backoff(
      [=]() -> bool { return !__cxx_nonatomic_compare_equal(__cxx_atomic_load(__ptr, memory_order_relaxed), __val); },
      __libcpp_timed_backoff_policy());
}

static void __libcpp_platform_wake_by_address(__cxx_atomic_contention_t const volatile*, bool) {}

#endif // __linux__

static constexpr size_t __libcpp_contention_table_size = (1 << 8); /* < there's no magic in this number */

struct alignas(64) /*  aim to avoid false sharing */ __libcpp_contention_table_entry {
  __cxx_atomic_contention_t __contention_state; // this is the waiter count
  __cxx_atomic_contention_t __platform_state;
  inline constexpr __libcpp_contention_table_entry() : __contention_state(0), __platform_state(0) {}

  __cxx_atomic_contention_t& __waiter_count() { return __contention_state; }
};

static __libcpp_contention_table_entry __libcpp_contention_table[__libcpp_contention_table_size];

static hash<void const volatile*> __libcpp_contention_hasher;

static __libcpp_contention_table_entry* __libcpp_contention_state(void const volatile* p) {
  return &__libcpp_contention_table[__libcpp_contention_hasher(p) & (__libcpp_contention_table_size - 1)];
}

/* Given an atomic to track contention and an atomic to actually wait on, which may be
   the same atomic, we try to detect contention to avoid spuriously calling the platform. */

static void __libcpp_contention_notify(__cxx_atomic_contention_t volatile* __waiter_count,
                                       __cxx_atomic_contention_t const volatile* __platform_state,
                                       bool __notify_one) {
  if (0 != __cxx_atomic_load(__waiter_count, memory_order_seq_cst))
    // We only call 'wake' if we consumed a contention bit here.
    __libcpp_platform_wake_by_address(__platform_state, __notify_one);
}
static void __libcpp_contention_wait(__cxx_atomic_contention_t volatile* __waiter_count,
                                     __cxx_atomic_contention_t const volatile* __platform_state,
                                     __cxx_contention_t __old_value) {
  __cxx_atomic_fetch_add(__waiter_count, __cxx_contention_t(1), memory_order_seq_cst);
  // We sleep as long as the monitored value hasn't changed.
  __libcpp_platform_wait_on_address(__platform_state, __old_value);
  __cxx_atomic_fetch_sub(__waiter_count, __cxx_contention_t(1), memory_order_release);
}

namespace {

_LIBCPP_EXPORTED_FROM_ABI const auto __get_contention_table_entry = &__libcpp_contention_state;

}

_LIBCPP_EXPORTED_FROM_ABI __atomic_waitable_contention_self::__atomic_waitable_contention_self(
    const volatile __cxx_atomic_contention_t* __contention_address)
    : __waiter_count_(&__get_contention_table_entry(__contention_address)->__waiter_count()),
      __platform_state_(__contention_address) {}

_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __atomic_waitable_contention_self::__monitor() {
  return __cxx_atomic_load(__platform_state_, memory_order_acquire);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_waitable_contention_self::__wait(__cxx_contention_t __old_value) {
  __libcpp_contention_wait(__waiter_count_, __platform_state_, __old_value);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_waitable_contention_self::__notify_one() {
  __libcpp_contention_notify(__waiter_count_, __platform_state_, true);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_waitable_contention_self::__notify_all() {
  __libcpp_contention_notify(__waiter_count_, __platform_state_, false);
}

_LIBCPP_EXPORTED_FROM_ABI __atomic_waitable_contention_global::__atomic_waitable_contention_global(
    const volatile void* __atomic_waitable_address) {
  auto const __entry = __get_contention_table_entry(__atomic_waitable_address);
  __waiter_count_    = &__entry->__waiter_count();
  __platform_state_  = &__entry->__platform_state;
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_waitable_contention_global::__notify_one() { __notify_all(); }
_LIBCPP_EXPORTED_FROM_ABI void __atomic_waitable_contention_global::__notify_all() {
  // The value sequence laundering happens on the next line below.
  __cxx_atomic_fetch_add(__platform_state_, __cxx_contention_t(1), memory_order_release);
  __libcpp_contention_notify(
      __waiter_count_, __platform_state_, false /* when laundering, we can't handle notify_one */);
}

// All below functions unused now
static __cxx_contention_t
__libcpp_contention_monitor_for_wait(__cxx_atomic_contention_t volatile* /*__waiter_count*/,
                                     __cxx_atomic_contention_t const volatile* __platform_state) {
  // We will monitor this value.
  return __cxx_atomic_load(__platform_state, memory_order_acquire);
}

/* When the incoming atomic is the wrong size for the platform wait size, need to
   launder the value sequence through an atomic from our table. */

static void __libcpp_atomic_notify(void const volatile* __location) {
  auto const __entry = __get_contention_table_entry(__location);
  // The value sequence laundering happens on the next line below.
  __cxx_atomic_fetch_add(&__entry->__platform_state, __cxx_contention_t(1), memory_order_release);
  __libcpp_contention_notify(
      &__entry->__waiter_count(), &__entry->__platform_state, false /* when laundering, we can't handle notify_one */);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile* __location) {
  __libcpp_atomic_notify(__location);
}
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile* __location) {
  __libcpp_atomic_notify(__location);
}
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile* __location) {
  auto const __entry = __libcpp_contention_state(__location);
  return __libcpp_contention_monitor_for_wait(&__entry->__waiter_count(), &__entry->__platform_state);
}
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile* __location, __cxx_contention_t __old_value) {
  auto const __entry = __libcpp_contention_state(__location);
  __libcpp_contention_wait(&__entry->__waiter_count(), &__entry->__platform_state, __old_value);
}

/* When the incoming atomic happens to be the platform wait size, we still need to use the
   table for the contention detection, but we can use the atomic directly for the wait. */

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile* __location) {
  __libcpp_contention_notify(&__libcpp_contention_state(__location)->__waiter_count(), __location, true);
}
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile* __location) {
  __libcpp_contention_notify(&__libcpp_contention_state(__location)->__waiter_count(), __location, false);
}
// This function is never used, but still exported for ABI compatibility.
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile* __location) {
  return __libcpp_contention_monitor_for_wait(&__libcpp_contention_state(__location)->__waiter_count(), __location);
}
_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile* __location, __cxx_contention_t __old_value) {
  __libcpp_contention_wait(&__libcpp_contention_state(__location)->__waiter_count(), __location, __old_value);
}

_LIBCPP_END_NAMESPACE_STD
