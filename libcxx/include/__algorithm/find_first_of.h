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
#include <__algorithm/find.h>
#include <__algorithm/find_if.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/desugars_to.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _ForwardIterator1 __find_first_of(
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardIterator2 __last2,
    _BinaryPredicate&& __pred) {
  for (; __first1 != __last1; ++__first1) {
#ifndef _LIBCPP_CXX03_LANG

    using _Ref1 = typename iterator_traits<_ForwardIterator1>::reference;
    using _Ref2 = typename iterator_traits<_ForwardIterator2>::reference;

    if constexpr (__desugars_to<__equal_tag, _BinaryPredicate, _Ref1, _Ref2>) {
      _ForwardIterator2 __found = std::find(__first2, __last2, *__first1);
      if (__found != __last2)
        return __first1;
    } else {
      _ForwardIterator2 __found = std::find_if(__first2, __last2, [&](auto&& __x) { return __pred(*__first1, __x); });
      if (__found != __last2)
        return __first1;
    }

#else

    _ForwardIterator2 __found = std::find_if(__first2, __last2, [&](auto&& __x) { return __pred(*__first1, __x); });
    if (__found != __last2)
      return __first1;

#endif
  }
  return __last1;
}

template <class _ForwardIterator1, class _ForwardIterator2, class _BinaryPredicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1,
    _ForwardIterator1 __last1,
    _ForwardIterator2 __first2,
    _ForwardIterator2 __last2,
    _BinaryPredicate __pred) {
  return std::__find_first_of(__first1, __last1, __first2, __last2, __pred);
}

template <class _ForwardIterator1, class _ForwardIterator2>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator1 find_first_of(
    _ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2, _ForwardIterator2 __last2) {
  return std::__find_first_of(__first1, __last1, __first2, __last2, __equal_to());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FIND_FIRST_OF_H
