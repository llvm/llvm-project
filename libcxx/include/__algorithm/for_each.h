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

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Sent, class _Func>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void __for_each(_InputIterator __first, _Sent __last, _Func& __f) {
  for (; __first != __last; ++__first)
    __f(*__first);
}

// __segment_processor handles the per-segment processing by applying the function object __func_ to each
// element within the segment.
template <class _Func>
struct __segment_processor {
  _Func& __func_;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __segment_processor(_Func& __f) : __func_(__f) {}

  template <class _SegmentedIterator>
  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
  operator()(typename __segmented_iterator_traits<_SegmentedIterator>::__local_iterator __lfirst,
             typename __segmented_iterator_traits<_SegmentedIterator>::__local_iterator __llast) {
    std::__for_each(__lfirst, __llast, __func_);
  }
};

template <class _SegmentedIterator,
          class _Function,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__for_each(_SegmentedIterator __first, _SegmentedIterator __last, _Function& __func) {
  std::__for_each_segment(__first, __last, std::__segment_processor<_Function>(__func));
}

template <class _InputIterator, class _Function>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Function
for_each(_InputIterator __first, _InputIterator __last, _Function __f) {
  std::__for_each(__first, __last, __f);
  return __f;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FOR_EACH_H
