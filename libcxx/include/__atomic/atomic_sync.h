//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_H

#include <__atomic/atomic_waitable_traits.h>
#include <__atomic/contention_t.h>
#include <__atomic/memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__chrono/duration.h>
#include <__config>
#include <__memory/addressof.h>
#include <__thread/poll_with_backoff.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_same.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20
#  if _LIBCPP_HAS_THREADS

#    if !_LIBCPP_AVAILABILITY_HAS_NEW_SYNC

// old dylib interface kept for backwards compatibility
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t __libcpp_atomic_monitor(void const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __libcpp_atomic_wait(void const volatile*, __cxx_contention_t) _NOEXCEPT;

_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_one(__cxx_atomic_contention_t const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void __cxx_atomic_notify_all(__cxx_atomic_contention_t const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor(__cxx_atomic_contention_t const volatile*) _NOEXCEPT;
_LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait(__cxx_atomic_contention_t const volatile*, __cxx_contention_t) _NOEXCEPT;
#    endif // !_LIBCPP_AVAILABILITY_HAS_NEW_SYNC

// new dylib interface

// return the global contention state's current value for the address
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__atomic_monitor_global(void const* __address) _NOEXCEPT;

// wait on the global contention state to be changed from the given value for the address
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_global_table(void const* __address, __cxx_contention_t __monitor_value) _NOEXCEPT;

// notify one waiter waiting on the global contention state for the address
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_global_table(void const*) _NOEXCEPT;

// notify all waiters waiting on the global contention state for the address
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_global_table(void const*) _NOEXCEPT;

// wait on the address directly with the native platform wait
template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__atomic_wait_native(void const* __address, void const* __old_value) _NOEXCEPT;

// notify one waiter waiting on the address directly with the native platform wait
template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_one_native(const void*) _NOEXCEPT;

// notify all waiters waiting on the address directly with the native platform wait
template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void __atomic_notify_all_native(const void*) _NOEXCEPT;

#    if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

template <class _AtomicWaitable, class _Poll>
struct __atomic_wait_backoff_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;

  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;
  using __value_type _LIBCPP_NODEBUG      = typename __waitable_traits::__value_type;

  _LIBCPP_HIDE_FROM_ABI __backoff_results operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(4)) {
      auto __contention_address = const_cast<const void*>(
          static_cast<const volatile void*>(__waitable_traits::__atomic_contention_address(__a_)));

      if constexpr (__has_native_atomic_wait<__value_type>) {
        auto __atomic_value = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return __backoff_results::__poll_success;
        std::__atomic_wait_native<sizeof(__value_type)>(__contention_address, std::addressof(__atomic_value));
      } else {
        __cxx_contention_t __monitor_val = std::__atomic_monitor_global(__contention_address);
        auto __atomic_value              = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return __backoff_results::__poll_success;
        std::__atomic_wait_global_table(__contention_address, __monitor_val);
      }
    } else {
    } // poll
    return __backoff_results::__continue_poll;
  }
};

#    else // _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

template <class _AtomicWaitable, class _Poll>
struct __atomic_wait_backoff_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;

  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;

  _LIBCPP_HIDE_FROM_ABI bool
  __update_monitor_val_and_poll(__cxx_atomic_contention_t const volatile*, __cxx_contention_t& __monitor_val) const {
    // In case the contention type happens to be __cxx_atomic_contention_t, i.e. __cxx_atomic_impl<int64_t>,
    // the platform wait is directly monitoring the atomic value itself.
    // `__poll_` takes the current value of the atomic as an in-out argument
    // to potentially modify it. After it returns, `__monitor` has a value
    // which can be safely waited on by `std::__libcpp_atomic_wait` without any
    // ABA style issues.
    __monitor_val = __waitable_traits::__atomic_load(__a_, __order_);
    return __poll_(__monitor_val);
  }

  _LIBCPP_HIDE_FROM_ABI bool
  __update_monitor_val_and_poll(void const volatile* __contention_address, __cxx_contention_t& __monitor_val) const {
    // In case the contention type is anything else, platform wait is monitoring a __cxx_atomic_contention_t
    // from the global pool, the monitor comes from __libcpp_atomic_monitor
    __monitor_val      = std::__libcpp_atomic_monitor(__contention_address);
    auto __current_val = __waitable_traits::__atomic_load(__a_, __order_);
    return __poll_(__current_val);
  }

  _LIBCPP_HIDE_FROM_ABI __backoff_results operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(4)) {
      auto __contention_address = __waitable_traits::__atomic_contention_address(__a_);
      __cxx_contention_t __monitor_val;
      if (__update_monitor_val_and_poll(__contention_address, __monitor_val))
        return __backoff_results::__poll_success;
      std::__libcpp_atomic_wait(__contention_address, __monitor_val);
    } else {
    } // poll
    return __backoff_results::__continue_poll;
  }
};

#    endif // _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

// The semantics of this function are similar to `atomic`'s
// `.wait(T old, std::memory_order order)`, but instead of having a hardcoded
// predicate (is the loaded value unequal to `old`?), the predicate function is
// specified as an argument. The loaded value is given as an in-out argument to
// the predicate. If the predicate function returns `true`,
// `__atomic_wait_unless` will return. If the predicate function returns
// `false`, it must set the argument to its current understanding of the atomic
// value. The predicate function must not return `false` spuriously.
template <class _AtomicWaitable, class _Poll>
_LIBCPP_HIDE_FROM_ABI void __atomic_wait_unless(const _AtomicWaitable& __a, memory_order __order, _Poll&& __poll) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  __atomic_wait_backoff_impl<_AtomicWaitable, __decay_t<_Poll> > __backoff_fn = {__a, __poll, __order};
  std::__libcpp_thread_poll_with_backoff(
      /* poll */
      [&]() {
        auto __current_val = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_load(__a, __order);
        return __poll(__current_val);
      },
      /* backoff */ __backoff_fn);
}

#    if _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  using __value_type _LIBCPP_NODEBUG = typename __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__value_type;
  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;
  auto __contention_address =
      const_cast<const void*>(static_cast<const volatile void*>(__waitable_traits::__atomic_contention_address(__a)));
  if constexpr (__has_native_atomic_wait<__value_type>) {
    std::__atomic_notify_one_native<sizeof(__value_type)>(__contention_address);
  } else {
    std::__atomic_notify_one_global_table(__contention_address);
  }
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  using __value_type _LIBCPP_NODEBUG = typename __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__value_type;
  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;
  auto __contention_address =
      const_cast<const void*>(static_cast<const volatile void*>(__waitable_traits::__atomic_contention_address(__a)));
  if constexpr (__has_native_atomic_wait<__value_type>) {
    std::__atomic_notify_all_native<sizeof(__value_type)>(__contention_address);
  } else {
    std::__atomic_notify_all_global_table(__contention_address);
  }
}

#    else // _LIBCPP_AVAILABILITY_HAS_NEW_SYNC

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  std::__cxx_atomic_notify_one(__atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  std::__cxx_atomic_notify_all(__atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
}

#    endif

#  else // _LIBCPP_HAS_THREADS

template <class _AtomicWaitable, class _Poll>
_LIBCPP_HIDE_FROM_ABI void __atomic_wait_unless(const _AtomicWaitable& __a, memory_order __order, _Poll&& __poll) {
  std::__libcpp_thread_poll_with_backoff(
      /* poll */
      [&]() {
        auto __current_val = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_load(__a, __order);
        return __poll(__current_val);
      },
      /* backoff */ __spinning_backoff_policy());
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable&) {}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable&) {}

#  endif // _LIBCPP_HAS_THREADS

template <typename _Tp>
_LIBCPP_HIDE_FROM_ABI bool __cxx_nonatomic_compare_equal(_Tp const& __lhs, _Tp const& __rhs) {
  return std::memcmp(std::addressof(__lhs), std::addressof(__rhs), sizeof(_Tp)) == 0;
}

template <class _AtomicWaitable, class _Tp>
_LIBCPP_HIDE_FROM_ABI void __atomic_wait(_AtomicWaitable& __a, _Tp __val, memory_order __order) {
  static_assert(__atomic_waitable<_AtomicWaitable>);
  std::__atomic_wait_unless(__a, __order, [&](_Tp const& __current) {
    return !std::__cxx_nonatomic_compare_equal(__current, __val);
  });
}

#endif // C++20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_H
