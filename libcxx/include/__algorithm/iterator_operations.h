//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORIHTM_ITERATOR_OPERATIONS_H
#define _LIBCPP___ALGORIHTM_ITERATOR_OPERATIONS_H

#include <__config>
#include <__iterator/advance.h>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER > 17 && !defined(_LIBCPP_HAS_NO_INCOMPLETE_RANGES)
struct _RangesIterOps {
  static constexpr auto advance = ranges::advance;
  static constexpr auto distance = ranges::distance;
  static constexpr auto next = ranges::next;
};
#endif

struct _StdIterOps {

  template <class _Iterator, class _Distance>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR_AFTER_CXX11 void advance(_Iterator& __iter, _Distance __count) {
    return std::advance(__iter, __count);
  }

  template <class _Iterator>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR_AFTER_CXX11
  typename iterator_traits<_Iterator>::difference_type distance(_Iterator __first, _Iterator __last) {
    return std::distance(__first, __last);
  }

  template <class _Iterator>
  _LIBCPP_HIDE_FROM_ABI static _LIBCPP_CONSTEXPR_AFTER_CXX11
  _Iterator next(_Iterator, _Iterator __last) {
    return __last;
  }

};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORIHTM_ITERATOR_OPERATIONS_H
