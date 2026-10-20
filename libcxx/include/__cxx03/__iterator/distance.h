// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_DISTANCE_H
#define _LIBCPP___CXX03___ITERATOR_DISTANCE_H

#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__type_traits/decay.h>
#include <__cxx03/__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIter>
inline _LIBCPP_HIDE_FROM_ABI typename iterator_traits<_InputIter>::difference_type
__distance(_InputIter __first, _InputIter __last, input_iterator_tag) {
  typename iterator_traits<_InputIter>::difference_type __r(0);
  for (; __first != __last; ++__first)
    ++__r;
  return __r;
}

template <class _RandIter>
inline _LIBCPP_HIDE_FROM_ABI typename iterator_traits<_RandIter>::difference_type
__distance(_RandIter __first, _RandIter __last, random_access_iterator_tag) {
  return __last - __first;
}

template <class _InputIter>
inline _LIBCPP_HIDE_FROM_ABI typename iterator_traits<_InputIter>::difference_type
distance(_InputIter __first, _InputIter __last) {
  return std::__distance(__first, __last, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_DISTANCE_H
