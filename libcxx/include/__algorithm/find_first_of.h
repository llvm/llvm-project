// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FIND_FIRST_OF_H
#define _LIBCPP___ALGORITHM_FIND_FIRST_OF_H

#include <__algorithm/comp.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ForwardIterator1 __find_first_of_ce(
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardIterator2 __last2,
    _BinaryPredicate&& __pred) {
  return std::find_if(__first1, __last1, [&](const auto& x) {
    return std::any_of(__first2, __last2, [&](const auto& y) { return __pred(x, y); });
  });
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardIterator2 __last2,
    _BinaryPredicate __pred) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  return std::__find_first_of_ce(__first1, __last1, __first2, __last2, __equal_to());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FIND_FIRST_OF_H
