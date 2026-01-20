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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <new>
#include <thread>
#include <type_traits>

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

#elif defined(_WIN32)

#  include <memory>
#  include <windows.h>

#else // <- Add other operating systems here

// Baseline needs no new headers

#  define _LIBCPP_FUTEX(...) syscall(SYS_futex, __VA_ARGS__)

#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef __linux__

template <std::size_t _Size>
static void __platform_wait_on_address(void const* __ptr, void const* __val, uint64_t __timeout_ns) {
  static_assert(_Size == 4, "Can only wait on 4 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  static constexpr timespec __default_timeout = {2, 0};
  timespec __timeout;
  if (__timeout_ns == 0) {
    __timeout = __default_timeout;
  } else {
    __timeout.tv_sec  = __timeout_ns / 1'000'000'000;
    __timeout.tv_nsec = __timeout_ns % 1'000'000'000;
  }
  _LIBCPP_FUTEX(__ptr, FUTEX_WAIT_PRIVATE, *reinterpret_cast<__cxx_contention_t const*>(&buffer), &__timeout, 0, 0);
}

template <std::size_t _Size>
static void __platform_wake_by_address(void const* __ptr, bool __notify_one) {
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
static void __platform_wait_on_address(void const* __ptr, void const* __val, uint64_t __timeout_ns) {
  static_assert(_Size == 8 || _Size == 4, "Can only wait on 8 bytes or 4 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  auto __timeout_us = __timeout_ns == 0 ? 0 : static_cast<uint32_t>(__timeout_ns / 1000);
  if constexpr (_Size == 4)
    __ulock_wait(
        UL_COMPARE_AND_WAIT, const_cast<void*>(__ptr), *reinterpret_cast<uint32_t const*>(&buffer), __timeout_us);
  else
    __ulock_wait(
        UL_COMPARE_AND_WAIT64, const_cast<void*>(__ptr), *reinterpret_cast<uint64_t const*>(&buffer), __timeout_us);
}

template <std::size_t _Size>
static void __platform_wake_by_address(void const* __ptr, bool __notify_one) {
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
static void __platform_wait_on_address(void const* __ptr, void const* __val, uint64_t __timeout_ns) {
  static_assert(_Size == 8, "Can only wait on 8 bytes value");
  char buffer[_Size];
  std::memcpy(&buffer, const_cast<const void*>(__val), _Size);
  if (__timeout_ns == 0) {
    _umtx_op(const_cast<void*>(__ptr), UMTX_OP_WAIT, *reinterpret_cast<__cxx_contention_t*>(&buffer), nullptr, nullptr);
  } else {
    _umtx_time ut;
    ut._timeout.tv_sec  = __timeout_ns / 1'000'000'000;
    ut._timeout.tv_nsec = __timeout_ns % 1'000'000'000;
    ut._flags           = 0;               // Relative time (not absolute)
    ut._clockid         = CLOCK_MONOTONIC; // Use monotonic clock

    _umtx_op(const_cast<void*>(__ptr),
             UMTX_OP_WAIT,
             *reinterpret_cast<__cxx_contention_t*>(&buffer),
             reinterpret_cast<void*>(sizeof(ut)), // Pass size as uaddr
             &ut);                                // Pass _umtx_time structure as uaddr2
  }
}

template <std::size_t _Size>
static void __platform_wake_by_address(void const* __ptr, bool __notify_one) {
  static_assert(_Size == 8, "Can only wake up on 8 bytes value");
  _umtx_op(const_cast<void*>(__ptr), UMTX_OP_WAKE, __notify_one ? 1 : INT_MAX, nullptr, nullptr);
}

#elif defined(_WIN32)

static void* win32_get_synch_api_function(const char* function_name) {
  // Attempt to load the API set. Note that as per the Microsoft STL implementation, we assume this API is already
  // loaded and accessible. While this isn't explicitly guaranteed by publicly available Win32 API documentation, it is
  // true in practice, and may be guaranteed by internal documentation not released publicly. In any case the fact that
  // the Microsoft STL made this assumption is reasonable basis to say that we can too. The alternative to this would be
  // to use LoadLibrary, but then leak the module handle. We can't call FreeLibrary, as this would have to be triggered
  // by a global static destructor, which would hang off DllMain, and calling FreeLibrary from DllMain is explicitly
  // mentioned as not being allowed:
  // https://learn.microsoft.com/en-us/windows/win32/dlls/dllmain
  // Given the range of bad options here, we have chosen to mirror what Microsoft did, as it seems fair to assume that
  // Microsoft will guarantee compatibility for us, as we are exposed to the same conditions as all existing Windows
  // apps using the Microsoft STL VS2015/2017/2019/2022 runtimes, where Windows 7 support has not been excluded at
  // compile time.
  static auto module_handle = GetModuleHandleW(L"api-ms-win-core-synch-l1-2-0.dll");
  if (module_handle == nullptr) {
    return nullptr;
  }

  // Attempt to locate the function in the API and return the result to the caller. Note that the NULL return from this
  // method is documented as being interchangeable with nullptr.
  // https://devblogs.microsoft.com/oldnewthing/20180307-00/?p=98175
  return reinterpret_cast<void*>(GetProcAddress(module_handle, function_name));
}

template <std::size_t _Size>
static void __platform_wait_on_address(void const* __ptr, void const* __val, uint64_t __timeout_ns) {
  static_assert(_Size == 8, "Can only wait on 8 bytes value");
  // WaitOnAddress was added in Windows 8 (build 9200)
  static auto wait_on_address =
      reinterpret_cast<BOOL(WINAPI*)(void*, PVOID, SIZE_T, DWORD)>(win32_get_synch_api_function("WaitOnAddress"));
  if (wait_on_address != nullptr) {
    wait_on_address(const_cast<void*>(__ptr),
                    const_cast<void*>(__val),
                    _Size,
                    __timeout_ns == 0 ? INFINITE : static_cast<DWORD>(__timeout_ns / 1'000'000));
  } else {
    __libcpp_thread_poll_with_backoff(
        [=]() -> bool { return std::memcmp(const_cast<const void*>(__ptr), __val, _Size) != 0; },
        __libcpp_timed_backoff_policy(),
        std::chrono::nanoseconds(__timeout_ns));
  }
}

template <std::size_t _Size>
static void __platform_wake_by_address(void const* __ptr, bool __notify_one) {
  static_assert(_Size == 8, "Can only wake up on 8 bytes value");
  if (__notify_one) {
    // WakeByAddressSingle was added in Windows 8 (build 9200)
    static auto wake_by_address_single =
        reinterpret_cast<void(WINAPI*)(PVOID)>(win32_get_synch_api_function("WakeByAddressSingle"));
    if (wake_by_address_single != nullptr) {
      wake_by_address_single(const_cast<void*>(__ptr));
    } else {
      // The fallback implementation of waking does nothing, as the fallback wait implementation just does polling, so
      // there's nothing to do here.
    }
  } else {
    // WakeByAddressAll was added in Windows 8 (build 9200)
    static auto wake_by_address_all =
        reinterpret_cast<void(WINAPI*)(PVOID)>(win32_get_synch_api_function("WakeByAddressAll"));
    if (wake_by_address_all != nullptr) {
      wake_by_address_all(const_cast<void*>(__ptr));
    } else {
      // The fallback implementation of waking does nothing, as the fallback wait implementation just does polling, so
      // there's nothing to do here.
    }
  }
}

#else // <- Add other operating systems here

// Baseline is just a timed backoff

template <std::size_t _Size>
static void __platform_wait_on_address(void const* __ptr, void const* __val, uint64_t __timeout_ns) {
  __libcpp_thread_poll_with_backoff(
      [=]() -> bool { return std::memcmp(const_cast<const void*>(__ptr), __val, _Size) != 0; },
      __libcpp_timed_backoff_policy(),
      std::chrono::nanoseconds(__timeout_ns));
}

template <std::size_t _Size>
static void __platform_wake_by_address(void const*, bool) {}

#endif // __linux__

// =============================
// Local hidden helper functions
// =============================

/* Given an atomic to track contention and an atomic to actually wait on, which may be
   the same atomic, we try to detect contention to avoid spuriously calling the platform. */

template <std::size_t _Size>
static void
__contention_notify(__cxx_atomic_contention_t* __waiter_count, void const* __address_to_notify, bool __notify_one) {
  if (0 != __cxx_atomic_load(__waiter_count, memory_order_seq_cst))
    // We only call 'wake' if we consumed a contention bit here.
    __platform_wake_by_address<_Size>(__address_to_notify, __notify_one);
}

template <std::size_t _Size>
static void __contention_wait(__cxx_atomic_contention_t* __waiter_count,
                              void const* __address_to_wait,
                              void const* __old_value,
                              uint64_t __timeout_ns) {
  __cxx_atomic_fetch_add(__waiter_count, __cxx_contention_t(1), memory_order_relaxed);
  // https://llvm.org/PR109290
  // There are no platform guarantees of a memory barrier in the platform wait implementation
  __cxx_atomic_thread_fence(memory_order_seq_cst);
  // We sleep as long as the monitored value hasn't changed.
  __platform_wait_on_address<_Size>(__address_to_wait, __old_value, __timeout_ns);
  __cxx_atomic_fetch_sub(__waiter_count, __cxx_contention_t(1), memory_order_release);
}

static constexpr size_t __contention_table_size = (1 << 8); /* < there's no magic in this number */

static constexpr hash<void const*> __contention_hasher;

// Waiter count table for all atomics with the correct size that use itself as the wait/notify address.

struct alignas(
    std::hardware_constructive_interference_size) /*  aim to avoid false sharing */ __contention_state_native {
  __cxx_atomic_contention_t __waiter_count;
  constexpr __contention_state_native() : __waiter_count(0) {}
};

static __contention_state_native __contention_table_native[__contention_table_size];

static __cxx_atomic_contention_t* __get_native_waiter_count(void const* p) {
  return &__contention_table_native[__contention_hasher(p) & (__contention_table_size - 1)].__waiter_count;
}

// Global contention table for all atomics with the wrong size that use the global table's atomic as wait/notify
// address.

struct alignas(
    std::hardware_constructive_interference_size) /*  aim to avoid false sharing */ __contention_state_global {
  __cxx_atomic_contention_t __waiter_count;
  __cxx_atomic_contention_t __platform_state;
  constexpr __contention_state_global() : __waiter_count(0), __platform_state(0) {}
};

static __contention_state_global __contention_table_global[__contention_table_size];

static __contention_state_global* __get_global_contention_state(void const* p) {
  return &__contention_table_global[__contention_hasher(p) & (__contention_table_size - 1)];
}

/* When the incoming atomic is the wrong size for the platform wait size, need to
   launder the value sequence through an atomic from our table. */

static void __atomic_notify_global_table(void const* __location) {
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
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __atomic_monitor_global(void const* __location) noexcept {
  auto const __entry = __get_global_contention_state(__location);
  return __cxx_atomic_load(&__entry->__platform_state, memory_order_acquire);
}

_LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_global_table(void const* __location, __cxx_contention_t __old_value) noexcept {
  auto const __entry = __get_global_contention_state(__location);
  __contention_wait<sizeof(__cxx_atomic_contention_t)>(
      &__entry->__waiter_count, &__entry->__platform_state, &__old_value, 0);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_global_table_with_timeout(
    void const* __location, __cxx_contention_t __old_value, uint64_t __timeout_ns) _NOEXCEPT {
  auto const __entry = __get_global_contention_state(__location);
  __contention_wait<sizeof(__cxx_atomic_contention_t)>(
      &__entry->__waiter_count, &__entry->__platform_state, &__old_value, __timeout_ns);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_global_table(void const* __location) noexcept {
  __atomic_notify_global_table(__location);
}

_LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_global_table(void const* __location) noexcept {
  __atomic_notify_global_table(__location);
}

// native

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_native(void const* __address, void const* __old_value) noexcept {
  __contention_wait<_Size>(__get_native_waiter_count(__address), __address, __old_value, 0);
}

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_native_with_timeout(void const* __address, void const* __old_value, uint64_t __timeout_ns) noexcept {
  __contention_wait<_Size>(__get_native_waiter_count(__address), __address, __old_value, __timeout_ns);
}

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_native(void const* __location) noexcept {
  __contention_notify<_Size>(__get_native_waiter_count(__location), __location, true);
}

template <std::size_t _Size>
_LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_native(void const* __location) noexcept {
  __contention_notify<_Size>(__get_native_waiter_count(__location), __location, false);
}

// ==================================================
// Instantiation of the templates with supported size
// ==================================================

#if defined(_LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE)

#  define _INSTANTIATE(_SIZE)                                                                                          \
    template _LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_native<_SIZE>(void const*, void const*) noexcept;            \
    template _LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_native_with_timeout<_SIZE>(                                  \
        void const*, void const*, uint64_t) noexcept;                                                                  \
    template _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_native<_SIZE>(void const*) noexcept;                   \
    template _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_native<_SIZE>(void const*) noexcept;

_LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_INSTANTIATE)

#  undef _INSTANTIATE

#else // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

template _LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_native<sizeof(__cxx_contention_t)>(void const* __address, void const* __old_value) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_native_with_timeout<sizeof(__cxx_contention_t)>(
    void const* __address, void const* __old_value, uint64_t) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void
__atomic_notify_one_native<sizeof(__cxx_contention_t)>(void const* __location) noexcept;

template _LIBCPP_EXPORTED_FROM_ABI void
__atomic_notify_all_native<sizeof(__cxx_contention_t)>(void const* __location) noexcept;

#endif // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

// =============================================================
// Old dylib exported symbols, for backwards compatibility
// =============================================================
_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wmissing-prototypes")

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile* __location) noexcept {
  __atomic_notify_global_table(const_cast<void const*>(__location));
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile* __location) noexcept {
  __atomic_notify_global_table(const_cast<void const*>(__location));
}

_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile* __location) noexcept {
  auto const __entry = __get_global_contention_state(const_cast<void const*>(__location));
  return __cxx_atomic_load(&__entry->__platform_state, memory_order_acquire);
}

_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(void const volatile* __location, __cxx_contention_t __old_value) noexcept {
  auto const __entry = __get_global_contention_state(const_cast<void const*>(__location));
  __contention_wait<sizeof(__cxx_atomic_contention_t)>(
      &__entry->__waiter_count, &__entry->__platform_state, &__old_value, 0);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile* __location) noexcept {
  auto __location_cast = const_cast<const void*>(static_cast<const volatile void*>(__location));
  __contention_notify<sizeof(__cxx_atomic_contention_t)>(
      __get_native_waiter_count(__location_cast), __location_cast, true);
}

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile* __location) noexcept {
  auto __location_cast = const_cast<const void*>(static_cast<const volatile void*>(__location));
  __contention_notify<sizeof(__cxx_atomic_contention_t)>(
      __get_native_waiter_count(__location_cast), __location_cast, false);
}

_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile* __location, __cxx_contention_t __old_value) noexcept {
  auto __location_cast = const_cast<const void*>(static_cast<const volatile void*>(__location));
  __contention_wait<sizeof(__cxx_atomic_contention_t)>(
      __get_native_waiter_count(__location_cast), __location_cast, &__old_value, 0);
}

// this function is even unused in the old ABI
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile* __location) noexcept {
  return __cxx_atomic_load(__location, memory_order_acquire);
}

_LIBCPP_DIAGNOSTIC_POP

_LIBCPP_END_NAMESPACE_STD
