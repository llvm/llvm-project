//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_COPY_BACKWARD_H
#define _LIBCPP___ALGORITHM_COPY_BACKWARD_H

#include <__algorithm/copy.h>
#include <__algorithm/unwrap_iter.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/reverse_iterator.h>
#include <cstring>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter1, class _Sent1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
pair<_Iter1, _Iter2> __copy_backward_impl(_Iter1 __first, _Sent1 __last, _Iter2 __result) {
  auto __ret = std::__copy(reverse_iterator<_Iter1>(__last),
                           reverse_iterator<_Sent1>(__first),
                           reverse_iterator<_Iter2>(__result));
  return pair<_Iter1, _Iter2>(__ret.first.base(), __ret.second.base());
}

template <class _Iter1, class _Sent1, class _Iter2>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
pair<_Iter1, _Iter2> __copy_backward(_Iter1 __first, _Sent1 __last, _Iter2 __result) {
  auto __ret = std::__copy_backward_impl(std::__unwrap_iter(__first),
                                         std::__unwrap_iter(__last),
                                         std::__unwrap_iter(__result));
  return pair<_Iter1, _Iter2>(std::__rewrap_iter(__first, __ret.first), std::__rewrap_iter(__result, __ret.second));
}

template <class _BidirectionalIterator1, class _BidirectionalIterator2>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX17
_BidirectionalIterator2
copy_backward(_BidirectionalIterator1 __first, _BidirectionalIterator1 __last, _BidirectionalIterator2 __result) {
  return std::__copy_backward(__first, __last, __result).second;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_COPY_BACKWARD_H
