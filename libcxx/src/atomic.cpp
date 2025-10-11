//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__atomic/contention_t.h>
#include <__thread/timed_backoff_policy.h>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstring>
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
#  define _LIBCPP_FUTEX(...) syscall(SYS_futex, __VA_ARGS__)

#elif defined(__FreeBSD__)

#  include <sys/types.h>
#  include <sys/umtx.h>

#  define _LIBCPP_FUTEX(...) syscall(SYS_futex, __VA_ARGS__)

#elif defined(__OpenBSD__)

#  include <sys/futex.h>

// OpenBSD has no indirect syscalls
#  define _LIBCPP_FUTEX(...) futex(__VA_ARGS__)

#else // <- Add other operating systems here

// Baseline needs no new headers

#  define _LIBCPP_FUTEX(...) syscall(SYS_futex, __VA_ARGS__)

#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef __linux__

template <std::size_t _Size>
static void __libcpp_platform_wait_on_address(void const volatile* __ptr, void const* __val) {
  static_assert(_Size == 4, "Can only wait on 4 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  static constexpr timespec __timeout = {2, 0};
  _LIBCPP_FUTEX(__ptr, FUTEX_WAIT_PRIVATE, *reinterpret_cast<__cxx_contention_t const*>(&buffer), &__timeout, 0, 0);
}

template <std::size_t _Size>
static void __libcpp_platform_wake_by_address(void const volatile* __ptr, bool __notify_one) {
  static_assert(_Size == 4, "Can only wake up on 4 bytes value");
  _LIBCPP_FUTEX(__ptr, FUTEX_WAKE_PRIVATE, __notify_one ? 1 : INT_MAX, 0, 0, 0);
}

#elif defined(__APPLE__) && defined(_LIBCPP_USE_ULOCK)

extern "C" int __ulock_wait(
    uint32_t operation, void* addr, uint64_t value, uint32_t timeout); /* timeout is specified in microseconds */
extern "C" int __ulock_wake(uint32_t operation, void* addr, uint64_t wake_value);

// https://github.com/apple/darwin-xnu/blob/2ff845c2e033bd0ff64b5b6aa6063a1f8f65aa32/bsd/sys/ulock.h#L82
#  define UL_COMPARE_AND_WAIT 1
#  define UL_COMPARE_AND_WAIT64 5
#  define ULF_WAKE_ALL 0x00000100

template <std::size_t _Size>
static void __libcpp_platform_wait_on_address(void const volatile* __ptr, void const* __val) {
  static_assert(_Size == 8 || _Size == 4, "Can only wait on 8 bytes or 4 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  if constexpr (_Size == 4)
    __ulock_wait(UL_COMPARE_AND_WAIT, const_cast<void*>(__ptr), *reinterpret_cast<uint32_t const*>(&buffer), 0);
  else
    __ulock_wait(UL_COMPARE_AND_WAIT64, const_cast<void*>(__ptr), *reinterpret_cast<uint64_t const*>(&buffer), 0);
}

template <std::size_t _Size>
static void __libcpp_platform_wake_by_address(void const volatile* __ptr, bool __notify_one) {
  static_assert(_Size == 8 || _Size == 4, "Can only wake up on 8 bytes or 4 bytes value");

  if constexpr (_Size == 4)
    __ulock_wake(UL_COMPARE_AND_WAIT | (__notify_one ? 0 : ULF_WAKE_ALL), const_cast<void*>(__ptr), 0);
  else
    __ulock_wake(UL_COMPARE_AND_WAIT64 | (__notify_one ? 0 : ULF_WAKE_ALL), const_cast<void*>(__ptr), 0);
}

#elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8
/*
 * Since __cxx_contention_t is int64_t even on 32bit FreeBSD
 * platforms, we have to use umtx ops that work on the long type, and
 * limit its use to architectures where long and int64_t are synonyms.
 */

template <std::size_t _Size>
static void __libcpp_platform_wait_on_address(void const volatile* __ptr, void const* __val) {
  static_assert(_Size == 8, "Can only wait on 8 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  _umtx_op(const_cast<void*>(__ptr), UMTX_OP_WAIT, *reinterpret_cast<__cxx_contention_t*>(&buffer), nullptr, nullptr);
}

template <std::size_t _Size>
static void __libcpp_platform_wake_by_address(void const volatile* __ptr, bool __notify_one) {
  static_assert(_Size == 8, "Can only wake up on 8 bytes value");
  _umtx_op(const_cast<void*>(__ptr), UMTX_OP_WAKE, __notify_one ? 1 : INT_MAX, nullptr, nullptr);
}

#else // <- Add other operating systems here

// Baseline is just a timed backoff

template <std::size_t _Size>
static void __libcpp_platform_wait_on_address(void const volatile* __ptr, void const* __val) {
  __libcpp_thread_poll_with_backoff(
      [=]() -> bool { return !std::memcmp(const_cast<const void*>(__ptr), __val, _Size); },
      __libcpp_timed_backoff_policy());
}

template <std::size_t _Size>
static void __libcpp_platform_wake_by_address(void const volatile*, bool) {}

#endif // __linux__

// =============================
// Local hidden helper functions
// =============================

/* Given an atomic to track contention and an atomic to actually wait on, which may be
   the same atomic, we try to detect contention to avoid spuriously calling the platform. */

template <std::size_t _Size>
static void __contention_notify(
    __cxx_atomic_contention_t volatile* __waiter_count, void const volatile* __address_to_notify, bool __notify_one) {
  if (0 != __cxx_atomic_load(__waiter_count, memory_order_seq_cst))
    // We only call 'wake' if we consumed a contention bit here.
    __libcpp_platform_wake_by_address<_Size>(__address_to_notify, __notify_one);
}

template <std::size_t _Size>
static void __contention_wait(__cxx_atomic_contention_t volatile* __waiter_count,
                              void const volatile* __address_to_wait,
                              void const* __old_value) {
  __cxx_atomic_fetch_add(__waiter_count, __cxx_contention_t(1), memory_order_relaxed);
  // https://llvm.org/PR109290
  // There are no platform guarantees of a memory barrier in the platform wait implementation
  __cxx_atomic_thread_fence(memory_order_seq_cst);
  // We sleep as long as the monitored value hasn't changed.
  __libcpp_platform_wait_on_address<_Size>(__address_to_wait, __old_value);
  __cxx_atomic_fetch_sub(__waiter_count, __cxx_contention_t(1), memory_order_release);
}

#if defined(__APPLE__) && defined(__aarch64__)
constexpr size_t __cache_line_size = 128;
#else
constexpr size_t __cache_line_size = 64;
#endif

static constexpr size_t __contention_table_size = (1 << 8); /* < there's no magic in this number */

static constexpr hash<void const volatile*> __contention_hasher;

// Waiter count table for all atomics with the correct size that use itself as the wait/notify address.

struct alignas(__cache_line_size) /*  aim to avoid false sharing */ __contention_state_native {
  __cxx_atomic_contention_t __waiter_count;
  constexpr __contention_state_native() : __waiter_count(0) {}
};

static __contention_state_native __contention_table_native[__contention_table_size];

static __cxx_atomic_contention_t* __get_native_waiter_count(void const volatile* p) {
  return &__contention_table_native[__contention_hasher(p) & (__contention_table_size - 1)].__waiter_count;
}

// Global contention table for all atomics with the wrong size that use the global table's atomic as wait/notify
// address.

struct alignas(__cache_line_size) /*  aim to avoid false sharing */ __contention_state_global {
  __cxx_atomic_contention_t __waiter_count;
  __cxx_atomic_contention_t __platform_state;
  constexpr __contention_state_global() : __waiter_count(0), __platform_state(0) {}
};

static __contention_state_global __contention_table_global[__contention_table_size];

static __contention_state_global* __get_global_contention_state(void const volatile* p) {
  return &__contention_table_global[__contention_hasher(p) & (__contention_table_size - 1)];
}

/* When the incoming atomic is the wrong size for the platform wait size, need to
   launder the value sequence through an atomic from our table. */

static void __atomic_notify_global_table(void const volatile* __location) {
  auto const __entry = __get_global_contention_state(__location);
  // The value sequence laundering happens on the next line below.
  __cxx_atomic_fetch_add(&__entry->__platform_state, __cxx_contention_t(1), memory_order_seq_cst);
  __contention_notify<sizeof(__cxx_atomic_contention_t)>(
      &__entry->__waiter_count, &__entry->__platform_state, false /* when laundering, we can't handle notify_one */);
}

// =============================
// New dylib exported symbols
// =============================

// global
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor_global(void const volatile* __location) noexcept {
  auto const __entry = __get_global_contention_state(__location);
  return __cxx_atomic_load(&__entry->__platform_state, memory_order_acquire);
}

_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_global_table(void const volatile* __location, __cxx_contention_t __old_value) noexcept {
  auto const __entry = __get_global_contention_state(__location);
  __contention_wait<sizeof(__cxx_atomic_contention_t)>(
      &__entry->__waiter_count, &__entry->__platform_state, &__old_value);
}

_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_global_table(void const volatile* __location) noexcept {
  __atomic_notify_global_table(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_global_table(void const volatile* __location) noexcept {
  __atomic_notify_global_table(__location);
}

// native

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native(void const volatile* __address, void const* __old_value) noexcept {
  __contention_wait<_Size>(__get_native_waiter_count(__address), __address, __old_value);
}

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native(void const volatile* __location) noexcept {
  __contention_notify<_Size>(__get_native_waiter_count(__location), __location, true);
}

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native(void const volatile* __location) noexcept {
  __contention_notify<_Size>(__get_native_waiter_count(__location), __location, false);
}

// Instantiation of the templates with supported size
#ifdef __linux__

template _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native<4>(void const volatile* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native<4>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native<4>(void const volatile* __location) noexcept;

#elif defined(__APPLE__) && defined(_LIBCPP_USE_ULOCK)

template _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native<4>(void const volatile* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native<8>(void const volatile* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native<4>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native<8>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native<4>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native<8>(void const volatile* __location) noexcept;

#elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8

template _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native<8>(void const volatile* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native<8>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native<8>(void const volatile* __location) noexcept;

#else // <- Add other operating systems here

template _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_native<8>(void const volatile* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_one_native<8>(void const volatile* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_notify_all_native<8>(void const volatile* __location) noexcept;

#endif // __linux__

// =============================================================
// Old dylib exported symbols, for backwards compatibility
// =============================================================

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile* __location) noexcept {
  __libcpp_atomic_notify_one_global_table(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile* __location) noexcept {
  __libcpp_atomic_notify_all_global_table(__location);
}

_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile* __location) noexcept {
  return __libcpp_atomic_monitor_global(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(void const volatile* __location, __cxx_contention_t __old_value) noexcept {
  __libcpp_atomic_wait_global_table(__location, __old_value);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile* __location) noexcept {
  __libcpp_atomic_notify_one_native<sizeof(__cxx_atomic_contention_t)>(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile* __location) noexcept {
  __libcpp_atomic_notify_all_native<sizeof(__cxx_atomic_contention_t)>(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile* __location, __cxx_contention_t __old_value) noexcept {
  __libcpp_atomic_wait_native<sizeof(__cxx_atomic_contention_t)>(__location, &__old_value);
}

// this function is even unused in the old ABI
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile* __location) noexcept {
  return __cxx_atomic_load(__location, memory_order_acquire);
}

_LIBCPP_END_NAMESPACE_STD
