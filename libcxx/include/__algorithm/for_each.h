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
#include <__functional/identity.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/invoke.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Sent, class _Func, class _Proj>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
__for_each(_InputIterator __first, _Sent __last, _Func& __f, _Proj& __proj) {
  for (; __first != __last; ++__first)
    std::__invoke(__f, std::__invoke(__proj, *__first));
  return __first;
}

// __segment_processor handles the per-segment processing by applying the function object __func_ to the
// projected value of each element within the segment. It serves as a functor utilized by the segmented
// iterator algorithms such as __for_each_segment and __for_each_n_segment.
template <class _SegmentedIterator, class _Func, class _Proj>
struct __segment_processor {
  using _Traits _LIBCPP_NODEBUG = __segmented_iterator_traits<_SegmentedIterator>;

  _Func& __func_;
  _Proj& __proj_;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR explicit __segment_processor(_Func& __f, _Proj& __p)
      : __func_(__f), __proj_(__p) {}

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
  operator()(typename _Traits::__local_iterator __lfirst, typename _Traits::__local_iterator __llast) {
    std::__for_each(__lfirst, __llast, __func_, __proj_);
  }
};

template <class _SegmentedIterator,
          class _Func,
          class _Proj,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _SegmentedIterator
__for_each(_SegmentedIterator __first, _SegmentedIterator __last, _Func& __f, _Proj& __p) {
  std::__for_each_segment(__first, __last, std::__segment_processor<_SegmentedIterator, _Func, _Proj>(__f, __p));
  return __last;
}

template <class _InputIterator, class _Func>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Func
for_each(_InputIterator __first, _InputIterator __last, _Func __f) {
  __identity __proj;
  std::__for_each(__first, __last, __f, __proj);
  return std::move(__f);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FOR_EACH_H
