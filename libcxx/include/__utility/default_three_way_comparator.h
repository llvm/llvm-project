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

template <class _Tp>
struct __default_three_way_comparator<_Tp, _Tp, __enable_if_t<is_arithmetic<_Tp>::value> > {
  _LIBCPP_HIDE_FROM_ABI static int operator()(_Tp __lhs, _Tp __rhs) {
    if (__lhs < __rhs)
      return -1;
    if (__lhs > __rhs)
      return 1;
    return 0;
  }
};

template <class _LHS, class _RHS, bool = true>
inline const bool __has_default_three_way_comparator_v = false;

template <class _LHS, class _RHS>
inline const bool
    __has_default_three_way_comparator_v< _LHS, _RHS, sizeof(__default_three_way_comparator<_LHS, _RHS>) >= 0> = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___UTILITY_DEFAULT_THREE_WAY_COMPARATOR_H
