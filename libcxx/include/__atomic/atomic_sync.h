//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_SYNC_H
#define _LIBCPP___ATOMIC_ATOMIC_SYNC_H

#include <__algorithm/ranges_find.h>
#include <__atomic/contention_t.h>
#include <__atomic/memory_order.h>
#include <__atomic/to_gcc_order.h>
#include <__chrono/duration.h>
#include <__config>
#include <__memory/addressof.h>
#include <__ranges/access.h>
#include <__thread/poll_with_backoff.h>
#include <__type_traits/conjunction.h>
#include <__type_traits/decay.h>
#include <__type_traits/has_unique_object_representation.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_same.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// The customisation points to enable the following functions:
// - __atomic_wait
// - __atomic_wait_unless
// - __atomic_notify_one
// - __atomic_notify_all
// Note that std::atomic<T>::wait was back-ported to C++03
// The below implementations look ugly to support C++03
template <class _Tp, class = void>
struct __atomic_waitable_traits {
  using __value_type _LIBCPP_NODEBUG = void;

  template <class _AtomicWaitable>
  static void __atomic_load(_AtomicWaitable&&, memory_order) = delete;

  template <class _AtomicWaitable>
  static void __atomic_contention_address(_AtomicWaitable&&) = delete;
};

template <class _Tp, class = void>
struct __atomic_waitable : false_type {};

template <class _Tp>
struct __atomic_waitable< _Tp,
                          __void_t<decltype(__atomic_waitable_traits<__decay_t<_Tp> >::__atomic_load(
                                       std::declval<const _Tp&>(), std::declval<memory_order>())),
                                   decltype(__atomic_waitable_traits<__decay_t<_Tp> >::__atomic_contention_address(
                                       std::declval<const _Tp&>()))> > : true_type {};

#if _LIBCPP_STD_VER >= 20
#  if _LIBCPP_HAS_THREADS

// old dylib interface
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

// new dylib interface
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI __cxx_contention_t
__libcpp_atomic_monitor_global(void const volatile* __address) _NOEXCEPT;

_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_wait_global_table(void const volatile* __address, __cxx_contention_t __monitor_value) _NOEXCEPT;

_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_notify_one_global_table(void const volatile*) _NOEXCEPT;
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_notify_all_global_table(void const volatile*) _NOEXCEPT;

template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void

__libcpp_atomic_wait_native(void const volatile* __address, void const* __old_value) _NOEXCEPT;
template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_notify_one_native(const volatile void*) _NOEXCEPT;

template <std::size_t _Size>
_LIBCPP_AVAILABILITY_NEW_SYNC _LIBCPP_EXPORTED_FROM_ABI void
__libcpp_atomic_notify_all_native(const volatile void*) _NOEXCEPT;

// concepts defines the types are supported natively by the platform's wait

#    if defined(_LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE)

#      ifdef __linux__

#        define _LIBCPP_ATOMIC_WAIT_SIZES_LIST(_APPLY) _APPLY(4)

#      elif defined(__APPLE__)

#        define _LIBCPP_ATOMIC_WAIT_SIZES_LIST(_APPLY)                                                                 \
          _APPLY(4)                                                                                                    \
          _APPLY(8)

#      elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8

#        define _LIBCPP_ATOMIC_WAIT_SIZES_LIST(_APPLY) _APPLY(8)

#      else

#        define _LIBCPP_ATOMIC_WAIT_SIZES_LIST(_APPLY) _APPLY(sizeof(__cxx_contention_t))

#      endif // __linux__

inline constexpr std::size_t __supported_native_wait_sizes[] = {
#      define _IDENTITY(_SIZE) _SIZE,
    _LIBCPP_ATOMIC_WAIT_SIZES_LIST(_IDENTITY)
#      undef _IDENTITY
};

template <class _Tp>
concept __atomic_wait_native_type =
    has_unique_object_representations_v<_Tp> &&
    std::ranges::find(__supported_native_wait_sizes, sizeof(_Tp)) != ranges::end(__supported_native_wait_sizes);

#    else // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

template <class _Tp>
concept __atomic_wait_native_type = is_same_v<_Tp, __cxx_contention_t>;

#    endif // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

template <class _AtomicWaitable, class _Poll>
struct __atomic_wait_backoff_impl {
  const _AtomicWaitable& __a_;
  _Poll __poll_;
  memory_order __order_;

  using __waitable_traits _LIBCPP_NODEBUG = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >;
  using __value_type _LIBCPP_NODEBUG      = typename __waitable_traits::__value_type;

  _LIBCPP_HIDE_FROM_ABI bool operator()(chrono::nanoseconds __elapsed) const {
    if (__elapsed > chrono::microseconds(4)) {
      auto __contention_address = __waitable_traits::__atomic_contention_address(__a_);

      if constexpr (__atomic_wait_native_type<__value_type>) {
        auto __atomic_value = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return true;
        std::__libcpp_atomic_wait_native<sizeof(__value_type)>(__contention_address, std::addressof(__atomic_value));
      } else {
        __cxx_contention_t __monitor_val = std::__libcpp_atomic_monitor_global(__contention_address);
        auto __atomic_value              = __waitable_traits::__atomic_load(__a_, __order_);
        if (__poll_(__atomic_value))
          return true;
        std::__libcpp_atomic_wait_global_table(__contention_address, __monitor_val);
      }
    } else {
    } // poll
    return false;
  }
};

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
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  __atomic_wait_backoff_impl<_AtomicWaitable, __decay_t<_Poll> > __backoff_fn = {__a, __poll, __order};
  std::__libcpp_thread_poll_with_backoff(
      /* poll */
      [&]() {
        auto __current_val = __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_load(__a, __order);
        return __poll(__current_val);
      },
      /* backoff */ __backoff_fn);
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_one(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  using __value_type _LIBCPP_NODEBUG = typename __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__value_type;
  if constexpr (__atomic_wait_native_type<__value_type>) {
    std::__libcpp_atomic_notify_one_native<sizeof(__value_type)>(
        __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
  } else {
    std::__libcpp_atomic_notify_one_global_table(
        __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
  }
}

template <class _AtomicWaitable>
_LIBCPP_HIDE_FROM_ABI void __atomic_notify_all(const _AtomicWaitable& __a) {
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  using __value_type _LIBCPP_NODEBUG = typename __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__value_type;
  if constexpr (__atomic_wait_native_type<__value_type>) {
    std::__libcpp_atomic_notify_all_native<sizeof(__value_type)>(
        __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
  } else {
    std::__libcpp_atomic_notify_all_global_table(
        __atomic_waitable_traits<__decay_t<_AtomicWaitable> >::__atomic_contention_address(__a));
  }
}

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
  static_assert(__atomic_waitable<_AtomicWaitable>::value, "");
  std::__atomic_wait_unless(__a, __order, [&](_Tp const& __current) {
    return !std::__cxx_nonatomic_compare_equal(__current, __val);
  });
}

#endif // C++20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_SYNC_H
