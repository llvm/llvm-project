// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ITERATOR_PREV_H
#define _LIBCPP___CXX03___ITERATOR_PREV_H

#include <__cxx03/__assert>
#include <__cxx03/__config>
#include <__cxx03/__iterator/advance.h>
#include <__cxx03/__iterator/iterator_traits.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIter, __enable_if_t<__has_input_iterator_category<_InputIter>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _InputIter
prev(_InputIter __x, typename iterator_traits<_InputIter>::difference_type __n) {
  // Calling `advance` with a negative value on a non-bidirectional iterator is a no-op in the current implementation.
  // Note that this check duplicates the similar check in `std::advance`.
  _LIBCPP_ASSERT_PEDANTIC(__n <= 0 || __has_bidirectional_iterator_category<_InputIter>::value,
                          "Attempt to prev(it, n) with a positive n on a non-bidirectional iterator");
  std::advance(__x, -__n);
  return __x;
}

// LWG 3197
// It is unclear what the implications of "BidirectionalIterator" in the standard are.
// However, calling std::prev(non-bidi-iterator) is obviously an error and we should catch it at compile time.
template <class _InputIter, __enable_if_t<__has_input_iterator_category<_InputIter>::value, int> = 0>
[[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI _InputIter prev(_InputIter __it) {
  static_assert(__has_bidirectional_iterator_category<_InputIter>::value,
                "Attempt to prev(it) with a non-bidirectional iterator");
  return std::prev(std::move(__it), 1);
}

_LIBCPP_POP_MACROS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___ITERATOR_PREV_H
