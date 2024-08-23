// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_COUNT_IF_H
#define _LIBCPP___ALGORITHM_COUNT_IF_H

#include <__algorithm/for_each.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator,
          class _Predicate,
          __enable_if_t<!__is_segmented_iterator<_InputIterator>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    typename iterator_traits<_InputIterator>::difference_type
    __count_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  typename iterator_traits<_InputIterator>::difference_type __r(0);
  for (; __first != __last; ++__first)
    if (__pred(*__first))
      ++__r;
  return __r;
}

template <class _SegmentedIterator,
          class _Predicate,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    typename iterator_traits<_SegmentedIterator>::difference_type
    __count_if(_SegmentedIterator __first, _SegmentedIterator __last, _Predicate __pred) {
  typename iterator_traits<_SegmentedIterator>::difference_type __r(0);
  std::for_each(__first, __last, [&__r, __pred](auto& __val) mutable {
    if (__pred(__val))
      ++__r;
  });
  return __r;
}

template <class _InputIterator, class _Predicate>
_LIBCPP_NODISCARD inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
    typename iterator_traits<_InputIterator>::difference_type
    count_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  return __count_if(__first, __last, __pred);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_COUNT_IF_H