//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___UTILITY_DEFAULT_THREE_WAY_COMPARATOR_H
#define _LIBCPP___UTILITY_DEFAULT_THREE_WAY_COMPARATOR_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_arithmetic.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// This struct can be specialized to provide a three way comparator between _LHS and _RHS.
// The return value should be
// - less than zero if (lhs_val < rhs_val)
// - greater than zero if (rhs_val < lhs_val)
// - zero otherwise
template <class _LHS, class _RHS, class = void>
struct __default_three_way_comparator;

template <class _LHS, class _RHS>
struct __default_three_way_comparator<_LHS,
                                      _RHS,
                                      __enable_if_t<is_arithmetic<_LHS>::value && is_arithmetic<_RHS>::value> > {
  _LIBCPP_HIDE_FROM_ABI static int operator()(_LHS __lhs, _RHS __rhs) {
    if (__lhs < __rhs)
      return -1;
    if (__lhs > __rhs)
      return 1;
    return 0;
  }
};

#if _LIBCPP_STD_VER >= 20 && __has_builtin(__builtin_lt_synthesises_from_spaceship)
template <class _LHS, class _RHS>
struct __default_three_way_comparator<
    _LHS,
    _RHS,
    __enable_if_t<!(is_arithmetic<_LHS>::value && is_arithmetic<_RHS>::value) &&
                  __builtin_lt_synthesises_from_spaceship(const _LHS&, const _RHS&)>> {
  _LIBCPP_HIDE_FROM_ABI static int operator()(const _LHS& __lhs, const _RHS& __rhs) {
    auto __res = __lhs <=> __rhs;
    if (__res < 0)
      return -1;
    if (__res > 0)
      return 1;
    return 0;
  }
};
#endif

template <class _LHS, class _RHS, bool = true>
struct __has_default_three_way_comparator : false_type {};

template <class _LHS, class _RHS>
struct __has_default_three_way_comparator<_LHS, _RHS, sizeof(__default_three_way_comparator<_LHS, _RHS>) >= 0>
    : true_type {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_DEFAULT_THREE_WAY_COMPARATOR_H
