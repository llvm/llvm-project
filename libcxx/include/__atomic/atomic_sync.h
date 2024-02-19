//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_H

#include <__atomic/contention_t.h>
#include <__atomic/cxx_atomic_impl.h>
#include <__atomic/memory_order.h>
#include <__availability>
#include <__chrono/duration.h>
#include <__config>
#include <__memory/addressof.h>
#include <__thread/poll_with_backoff.h>
#include <__thread/support.h>
#include <__type_traits/decay.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Atp, class _Poll>
struct __libcpp_atomic_wait_poll_impl {
  _Atp* __a_;
  _Poll __poll_;
  memory_order __order_;

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool operator()() const {
    auto __current_val = std::__cxx_atomic_load(__a_, __order_);
    return __poll_(__current_val);
  }
};

#ifndef _LIBCPP_HAS_NO_THREADS

_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile*, __cxx_contention_t);

_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile*);
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile*, __cxx_contention_t);

template <class _Atp, class _Poll>
struct __libcpp_atomic_wait_backoff_impl {
  _Atp* __a_;
  _Poll __poll_;
  memory_order __order_;

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool
  __poll_or_get_monitor(__cxx_atomic_contention_t const volatile*, __cxx_contention_t& __monitor) const {
    // In case the atomic can be waited on directly, the monitor value is just
    // the value of the atomic.
    // `__poll_` takes the current value of the atomic as an in-out argument
    // to potentially modify it. After it returns, `__monitor` has a value
    // which can be safely waited on by `std::__libcpp_atomic_wait` without any
    // ABA style issues.
    __monitor = std::__cxx_atomic_load(__a_, __order_);
    return __poll_(__monitor);
  }

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool __poll_or_get_monitor(void const volatile*, __cxx_contention_t& __monitor) const {
    // In case we must wait on an atomic from the pool, the monitor comes from
    // `std::__libcpp_atomic_monitor`.
    // Only then we may read from `__a_`. This is the "event count" pattern.
    __monitor          = std::__libcpp_atomic_monitor(__a_);
    auto __current_val = std::__cxx_atomic_load(__a_, __order_);
    return __poll_(__current_val);
  }

  _LIBCPP_AVAILABILITY_SYNC
  _LIBCPP_HIDE_FROM_ABI bool operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(64)) {
      __cxx_contention_t __monitor;
      if (__poll_or_get_monitor(__a_, __monitor))
        return true;
      std::__libcpp_atomic_wait(__a_, __monitor);
    } else if (__elapsed > chrono::microseconds(4))
      __libcpp_thread_yield();
    else {
    } // poll
    return false;
  }
};

// The semantics of this function are similar to `atomic`'s
// `.wait(T old, std::memory_order order)`, but instead of having a hardcoded
// predicate (is the loaded value unequal to `old`?), the predicate function is
// specified as an argument. The loaded value is given as an in-out argument to
// the predicate. If the predicate function returns `true`,
// `_cxx_atomic_wait_unless` will return. If the predicate function returns
// `false`, it must set the argument to its current understanding of the atomic
// value. The predicate function must not return `false` spuriously.
template <class _Atp, class _Poll>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void
__cxx_atomic_wait_unless(_Atp* __a, _Poll&& __poll, memory_order __order) {
  __libcpp_atomic_wait_poll_impl<_Atp, __decay_t<_Poll> > __poll_fn       = {__a, __poll, __order};
  __libcpp_atomic_wait_backoff_impl<_Atp, __decay_t<_Poll> > __backoff_fn = {__a, __poll, __order};
  (void)std::__libcpp_thread_poll_with_backoff(__poll_fn, __backoff_fn);
}

#else // _LIBCPP_HAS_NO_THREADS

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI void __cxx_atomic_notify_all(__cxx_atomic_impl<_Tp> const volatile*) {}
template <class _Tp>
_LIBCPP_HIDE_FROM_ABI void __cxx_atomic_notify_one(__cxx_atomic_impl<_Tp> const volatile*) {}
template <class _Atp, class _Poll>
_LIBCPP_HIDE_FROM_ABI void __cxx_atomic_wait_unless(_Atp* __a, _Poll&& __poll, memory_order __order) {
  __libcpp_atomic_wait_poll_impl<_Atp, __decay_t<_Poll> > __poll_fn = {__a, __poll, __order};
  (void)std::__libcpp_thread_poll_with_backoff(__poll_fn, __spinning_backoff_policy());
}

#endif // _LIBCPP_HAS_NO_THREADS

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI bool __cxx_nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs) {
  return std::memcmp(std::addressof(__lhs), std::addressof(__rhs), sizeof(_Tp)) == 0;
}

template <class _Tp>
struct __atomic_compare_unequal_to {
  _Tp __val;
  _LIBCPP_HIDE_FROM_ABI bool operator()(_Tp& __current_val) const {
    return !std::__cxx_nonatomic_compare_equal(__current_val, __val);
  }
};

template <class _Atp, class _Tp>
_LIBCPP_AVAILABILITY_SYNC _LIBCPP_HIDE_FROM_ABI void
__cxx_atomic_wait(_Atp* __a, _Tp const __val, memory_order __order) {
  __atomic_compare_unequal_to<_Tp> __poll_fn = {__val};
  std::__cxx_atomic_wait_unless(__a, __poll_fn, __order);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_H
