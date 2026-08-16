//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___COMPARE_TYPE_ORDER
#define _LIBCPP___COMPARE_TYPE_ORDER

#include <__compare/ordering.h>
#include <__config>

#ifndef _LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 26 && __has_builtin(__builtin_type_order)

// [compare.type], type ordering
template <class _Tp, class _Up>
struct _LIBCPP_NO_SPECIALIZATIONS type_order {
  static constexpr strong_ordering value = __builtin_type_order(_Tp, _Up);
  using value_type                       = strong_ordering;

  _LIBCPP_HIDE_FROM_ABI constexpr operator value_type() const noexcept { return value; }
  _LIBCPP_HIDE_FROM_ABI constexpr value_type operator()() const noexcept { return value; }
};

template <class _Tp, class _Up>
constexpr strong_ordering type_order_v = __builtin_type_order(_Tp, _Up);

#endif // _LIBCPP_STD_VER >= 26 && __has_builtin(__builtin_type_order)

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___COMPARE_TYPE_ORDER
