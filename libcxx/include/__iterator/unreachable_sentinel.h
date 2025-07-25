// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ITERATOR_UNREACHABLE_SENTINEL_H
#define _LIBCPP___ITERATOR_UNREACHABLE_SENTINEL_H

#include <__config>
#include <__iterator/concepts.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_same.h>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

inline constexpr struct __unreachable_sentinel_t {} __unreachable_sentinel;

template <class _UnreachableSentinel,
          class _Iter,
          __enable_if_t<is_same<__remove_cvref_t<_UnreachableSentinel>, __unreachable_sentinel_t>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(_UnreachableSentinel&&, _Iter&&) {
  return false;
}

template <class _UnreachableSentinel,
          class _Iter,
          __enable_if_t<is_same<__remove_cvref_t<_UnreachableSentinel>, __unreachable_sentinel_t>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator==(_Iter&&, _UnreachableSentinel&&) {
  return false;
}

template <class _UnreachableSentinel,
          class _Iter,
          __enable_if_t<is_same<__remove_cvref_t<_UnreachableSentinel>, __unreachable_sentinel_t>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(_UnreachableSentinel&&, _Iter&&) {
  return true;
}

template <class _UnreachableSentinel,
          class _Iter,
          __enable_if_t<is_same<__remove_cvref_t<_UnreachableSentinel>, __unreachable_sentinel_t>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI constexpr bool operator!=(_Iter&&, _UnreachableSentinel&&) {
  return true;
}

#if _LIBCPP_STD_VER >= 20

struct unreachable_sentinel_t {
  template <weakly_incrementable _Iter>
  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(unreachable_sentinel_t, const _Iter&) noexcept {
    return false;
  }
};

inline constexpr unreachable_sentinel_t unreachable_sentinel{};

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ITERATOR_UNREACHABLE_SENTINEL_H
