// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOR_EACH_H
#define _LIBCPP___ALGORITHM_FOR_EACH_H

#include <__algorithm/for_each_segment.h>
#include <__config>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>
#include <__utility/in_place.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Sent, class _Func>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __for_each(_InputIterator __first, _Sent __last, _Func& __f) {
  for (; __first != __last; ++__first)
    __f(*__first);
}

#ifndef _LIBCPP_CXX03_LANG
template <class _SegmentedIterator,
          class _Function,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__for_each(_SegmentedIterator __first, _SegmentedIterator __last, _Function& __func) {
  using _Traits = __segmented_iterator_traits<_SegmentedIterator>;
  std::__for_each_segment(
      __first, __last, [&](typename _Traits::__local_iterator __lfirst, typename _Traits::__local_iterator __llast) {
        std::__for_each(__lfirst, __llast, __func);
      });
}
#endif

template <class _InputIterator, class _Function>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Function
for_each(_InputIterator __first, _InputIterator __last, _Function __f) {
  std::__for_each(__first, __last, __f);
  return __f;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOR_EACH_H
