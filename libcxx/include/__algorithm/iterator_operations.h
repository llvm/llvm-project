//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_ITERATOR_OPERATIONS_H
#define _LIBCPP___ALGORITHM_ITERATOR_OPERATIONS_H

#include <__algorithm/iter_swap.h>
#include <__config>
#include <__iterator/advance.h>
#include <__iterator/distance.h>
#include <__iterator/iter_move.h>
#include <__iterator/iter_swap.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__utility/forward.h>
#include <__utility/move.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy> struct _IterOps;

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)
struct _RangeAlgPolicy {};

template <>
struct _IterOps<_RangeAlgPolicy> {
  static constexpr auto advance = ranges::advance;
  static constexpr auto distance = ranges::distance;
  static constexpr auto __iter_move = ranges::iter_move;
  static constexpr auto iter_swap = ranges::iter_swap;
  static constexpr auto next = ranges::next;
  static constexpr auto __advance_to = ranges::advance;
};

#endif

struct _ClassicAlgPolicy {};

template <>
struct _IterOps<_ClassicAlgPolicy> {

  // advance
  template <class _Iter, class _Distance>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
  static void advance(_Iter& __iter, _Distance __count) {
    std::advance(__iter, __count);
  }

  // distance
  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
  static typename iterator_traits<_Iter>::difference_type distance(_Iter __first, _Iter __last) {
    return std::distance(__first, __last);
  }

  // iter_move
  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
  // Declaring the return type is necessary for the C++03 mode (which doesn't support placeholder return types).
  static typename iterator_traits<__uncvref_t<_Iter> >::value_type&& __iter_move(_Iter&& __i) {
    return std::move(*std::forward<_Iter>(__i));
  }

  // iter_swap
  template <class _Iter1, class _Iter2>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_AFTER_CXX11
  static void iter_swap(_Iter1&& __a, _Iter2&& __b) {
    std::iter_swap(std::forward<_Iter1>(__a), std::forward<_Iter2>(__b));
  }

  // next
  template <class _Iterator>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR_AFTER_CXX11
  _Iterator next(_Iterator, _Iterator __last) {
    return __last;
  }

  template <class _Iter>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR_AFTER_CXX11
  void __advance_to(_Iter& __first, _Iter __last) {
    __first = __last;
  }
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_ITERATOR_OPERATIONS_H
