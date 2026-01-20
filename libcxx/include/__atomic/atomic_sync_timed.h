//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_TIMED_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_TIMED_H

#include <__atomic/atomic_waitable_traits.h>
#include <__atomic/contention_t.h>
#include <__atomic/memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__chrono/duration.h>
#include <__config>
#include <__memory/addressof.h>
#include <__thread/poll_with_backoff.h>
#include <__thread/timed_backoff_policy.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/has_unique_object_representation.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>
#include <cstdint>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20
#  if _LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

_LIBCPP_AVAILABILITY_NEW_SYNC
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __atomic_monitor_global(void const* __address) _NOEXCEPT;

// wait on the global contention state to be changed from the given value for the address
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_wait_global_table_with_timeout(
    void const* __address, __cxx_contention_t __monitor_value, uint64_t __timeout_ns) _NOEXCEPT;

// wait on the address directly with the native platform wait
template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_native_with_timeout(void const* __address, void const* __old_value, uint64_t __timeout_ns) _NOEXCEPT;

template <class _AtomicWaitable, class _Poll, class _Rep, class _Period>
struct __atomic_wait_timed_backoff_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;
  chrono::duration<_Rep, _Period> __rel_time_;

  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;
  using __value_type _LIBCPP_NODEBUG      = typename __waitable_traits::__value_type;

  _LIBCPP_HIDE_FROM_ABI __backoff_results operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(4)) {
      auto __contention_address = const_cast<const void*>(
          static_cast<const volatile void*>(__waitable_traits::__atomic_contention_address(__a_)));

      uint64_t __timeout_ns =
          static_cast<uint64_t>((chrono::duration_cast<chrono::nanoseconds>(__rel_time_) - __elapsed).count());

      if constexpr (__has_native_atomic_wait<__value_type>) {
        auto __atomic_value = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return __backoff_results::__poll_success;
        std::__atomic_wait_native_with_timeout<sizeof(__value_type)>(
            __contention_address, std::addressof(__atomic_value), __timeout_ns);
      } else {
        __cxx_contention_t __monitor_val = std::__atomic_monitor_global(__contention_address);
        auto __atomic_value              = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return __backoff_results::__poll_success;
        std::__atomic_wait_global_table_with_timeout(__contention_address, __monitor_val, __timeout_ns);
      }
    } else {
    } // poll
    return __backoff_results::__continue_poll;
  }
};

// The semantics of this function are similar to `atomic`'s
// `.wait(T old, std::memory_order order)` with a timeout, but instead of having a hardcoded
// predicate (is the loaded value unequal to `old`?), the predicate function is
// specified as an argument. The loaded value is given as an in-out argument to
// the predicate. If the predicate function returns `true`,
// `__atomic_wait_unless_with_timeout` will return. If the predicate function returns
// `false`, it must set the argument to its current understanding of the atomic
// value. The predicate function must not return `false` spuriously.
template <class _AtomicWaitable, class _Poll, class _Rep, class _Period>
_LIBCPP_HIDE_FROM_ABI bool __atomic_wait_unless_with_timeout(
    const _AtomicWaitable& __a,
    memory_order __order,
    _Poll&& __poll,
    chrono::duration<_Rep, _Period> const& __rel_time) {
  static_assert(__atomic_waitable<_AtomicWaitable>, "");
  __atomic_wait_timed_backoff_impl<_AtomicWaitable, __decay_t<_Poll>, _Rep, _Period> __backoff_fn = {
      __a, __poll, __order, __rel_time};
  auto __poll_result = std::__libcpp_thread_poll_with_backoff(
      /* poll */
      [&]() {
        auto __current_val = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_load(__a, __order);
        return __poll(__current_val);
      },
      /* backoff */ __backoff_fn,
      __rel_time);

  return __poll_result == __poll_with_backoff_results::__poll_success;
}

#  elif _LIBCPP_HAS_THREADS // _LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

template <class _AtomicWaitable, class _Poll, class _Rep, class _Period>
_LIBCPP_HIDE_FROM_ABI bool __atomic_wait_unless_with_timeout(
    const _AtomicWaitable& __a,
    memory_order __order,
    _Poll&& __poll,
    chrono::duration<_Rep, _Period> const& __rel_time) {
  auto __res = std::__libcpp_thread_poll_with_backoff(
      /* poll */
      [&]() {
        auto __current_val = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_load(__a, __order);
        return __poll(__current_val);
      },
      /* backoff */ __libcpp_timed_backoff_policy(),
      __rel_time);
  return __res == __poll_with_backoff_results::__poll_success;
}

#  endif // _LIBCPP_HAS_THREADS && _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

#endif // C++20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_TIMED_H
