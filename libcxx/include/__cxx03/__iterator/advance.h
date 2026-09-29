// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_ADVANCE_H
#define _LIBCPP___CXX03___ITERATOR_ADVANCE_H

#include <__cxx03/__assert>
#include <__cxx03/__config>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__type_traits/is_integral.h>
#include <__cxx03/__utility/convert_to_integral.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/__utility/unreachable.h>
#include <__cxx03/limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIter>
_LIBCPP_HIDE_FROM_ABI void
__advance(_InputIter& __i, typename iterator_traits<_InputIter>::difference_type __n, input_iterator_tag) {
  for (; __n > 0; --__n)
    ++__i;
}

template <class _BiDirIter>
_LIBCPP_HIDE_FROM_ABI void
__advance(_BiDirIter& __i, typename iterator_traits<_BiDirIter>::difference_type __n, bidirectional_iterator_tag) {
  if (__n >= 0)
    for (; __n > 0; --__n)
      ++__i;
  else
    for (; __n < 0; ++__n)
      --__i;
}

template <class _RandIter>
_LIBCPP_HIDE_FROM_ABI void
__advance(_RandIter& __i, typename iterator_traits<_RandIter>::difference_type __n, random_access_iterator_tag) {
  __i += __n;
}

template < class _InputIter,
           class _Distance,
           class _IntegralDistance = decltype(std::__convert_to_integral(std::declval<_Distance>())),
           __enable_if_t<is_integral<_IntegralDistance>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI void advance(_InputIter& __i, _Distance __orig_n) {
  typedef typename iterator_traits<_InputIter>::difference_type _Difference;
  _Difference __n = static_cast<_Difference>(std::__convert_to_integral(__orig_n));
  // Calling `advance` with a negative value on a non-bidirectional iterator is a no-op in the current implementation.
  _LIBCPP_ASSERT_PEDANTIC(__n >= 0 || __has_bidirectional_iterator_category<_InputIter>::value,
                          "Attempt to advance(it, n) with negative n on a non-bidirectional iterator");
  std::__advance(__i, __n, typename iterator_traits<_InputIter>::iterator_category());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ITERATOR_ADVANCE_H
