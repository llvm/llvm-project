//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_INCLUDES_H
#define _LIBCPP___ALGORITHM_INCLUDES_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter1, class _Sent1, class _Iter2, class _Sent2, class _Comp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX17 bool
__includes(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2, _Sent2 __last2, _Comp&& __comp) {
  for (; __first2 != __last2; ++__first1) {
    if (__first1 == __last1 || __comp(*__first2, *__first1))
      return false;
    if (!__comp(*__first1, *__first2))
      ++__first2;
  }
  return true;
}

template <class _InputIterator1, class _InputIterator2, class _Compare>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX17 bool includes(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _InputIterator2 __last2,
    _Compare __comp) {
  typedef typename __comp_ref_type<_Compare>::type _Comp_ref;
  return std::__includes(
      std::move(__first1), std::move(__last1), std::move(__first2), std::move(__last2), static_cast<_Comp_ref>(__comp));
}

template <class _InputIterator1, class _InputIterator2>
_LIBCPP_NODISCARD_EXT inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX17 bool
includes(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _InputIterator2 __last2) {
  return std::includes(
      std::move(__first1),
      std::move(__last1),
      std::move(__first2),
      std::move(__last2),
      __less<typename iterator_traits<_InputIterator1>::value_type,
             typename iterator_traits<_InputIterator2>::value_type>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_INCLUDES_H
