//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ATOMIC_WAITABLE_TRAITS_H
#define _LIBCPP___ATOMIC_ATOMIC_WAITABLE_TRAITS_H

#include <__atomic/contention_t.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__type_traits/decay.h>
#include <__type_traits/has_unique_object_representation.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_copyable.h>
#include <__type_traits/void_t.h>
#include <__utility/declval.h>
#include <cstring>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

// The customisation points to enable the following functions:
// - __atomic_wait
// - __atomic_wait_unless
// - __atomic_notify_one
// - __atomic_notify_all
template <class _Tp, class = void>
struct __atomic_waitable_traits {
  using __value_type _LIBCPP_NODEBUG = void;

  template <class _AtomicWaitable>
  static void __atomic_load(_AtomicWaitable&&, memory_order) = delete;

  template <class _AtomicWaitable>
  static void __atomic_contention_address(_AtomicWaitable&&) = delete;
};

template <class _Tp>
concept __atomic_waitable = requires(const _Tp __t, memory_order __order) {
  typename __atomic_waitable_traits<__decay_t<_Tp> >::__value_type;
  { __atomic_waitable_traits<__decay_t<_Tp> >::__atomic_load(__t, __order) };
  { __atomic_waitable_traits<__decay_t<_Tp> >::__atomic_contention_address(__t) };
};

#  ifdef __linux__
#    define _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_APPLY) _APPLY(4)
#  elif defined(__APPLE__)
#    define _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_APPLY)                                                                 \
      _APPLY(4)                                                                                                        \
      _APPLY(8)
#  elif defined(__FreeBSD__) && __SIZEOF_LONG__ == 8
#    define _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_APPLY) _APPLY(8)
#  elif defined(_WIN32)
#    define _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_APPLY) _APPLY(8)
#  else
#    define _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_APPLY) _APPLY(sizeof(__cxx_contention_t))
#  endif // __linux__

// concepts defines the types are supported natively by the platform's wait

#  if defined(_LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE)

_LIBCPP_HIDE_FROM_ABI constexpr bool __has_native_atomic_wait_impl(size_t __size) {
  switch (__size) {
#    define _LIBCPP_MAKE_CASE(n)                                                                                       \
    case n:                                                                                                            \
      return true;
    _LIBCPP_NATIVE_PLATFORM_WAIT_SIZES(_LIBCPP_MAKE_CASE)
  default:
    return false;
#    undef _LIBCPP_MAKE_CASE
  };
}

template <class _Tp>
concept __has_native_atomic_wait =
    has_unique_object_representations_v<_Tp> && is_trivially_copyable_v<_Tp> &&
    __has_native_atomic_wait_impl(sizeof(_Tp));

#  else // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

template <class _Tp>
concept __has_native_atomic_wait = is_same_v<_Tp, __cxx_contention_t>;

#  endif // _LIBCPP_ABI_ATOMIC_WAIT_NATIVE_BY_SIZE

#endif // C++20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ATOMIC_ATOMIC_WAITABLE_TRAITS_H
