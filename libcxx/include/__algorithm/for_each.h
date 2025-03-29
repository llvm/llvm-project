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
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Sent, class _Function>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Function
__for_each(_InputIterator __first, _Sent __last, _Function& __f) {
  for (; __first != __last; ++__first)
    __f(*__first);
  return std::move(__f);
}

// __do_segment acts as a functor for processing individual segments within the __for_each_segment{, _n} algorithms.
template <class _InputIterator, class _Function>
struct __do_segment {
  using _Traits _LIBCPP_NODEBUG = __segmented_iterator_traits<_InputIterator>;

  _Function& __func_;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 explicit __do_segment(_Function& __func) : __func_(__func) {}

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void
  operator()(typename _Traits::__local_iterator __lfirst, typename _Traits::__local_iterator __llast) {
    std::__for_each(__lfirst, __llast, __func_);
  }
};

template <class _SegmentedIterator,
          class _Function,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _Function
__for_each(_SegmentedIterator __first, _SegmentedIterator __last, _Function& __func) {
  std::__for_each_segment(__first, __last, std::__do_segment<_SegmentedIterator, _Function>(__func));
  return std::move(__func);
}

template <class _InputIterator, class _Function>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Function
for_each(_InputIterator __first, _InputIterator __last, _Function __f) {
  return std::__for_each(__first, __last, __f);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FOR_EACH_H
