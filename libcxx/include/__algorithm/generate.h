//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_GENERATE_H
#define _LIBCPP___ALGORITHM_GENERATE_H

#include <__algorithm/for_each_segment.h>
#include <__config>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator, class _Sent, class _Generator>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__generate(_ForwardIterator __first, _Sent __last, _Generator __gen) {
  for (; __first != __last; ++__first)
    *__first = __gen();
}

#ifndef _LIBCPP_CXX03_LANG
template <class _SegmentedIterator,
          class _Generator,
          __enable_if_t<__is_segmented_iterator_v<_SegmentedIterator>, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20
_SegmentedIterator __generate(_SegmentedIterator __first, _SegmentedIterator __last, _Generator& __gen) {
  using __local_iterator_t = typename __segmented_iterator_traits<_SegmentedIterator>::__local_iterator;
  std::__for_each_segment(__first, __last, [&](__local_iterator_t __lfirst, __local_iterator_t __llast) {
    std::__generate(__lfirst, __llast, __gen);
  });
  return __last;
}
#endif // !_LIBCPP_CXX03_LANG

template <class _ForwardIterator, class _Generator>
inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_CONSTEXPR_SINCE_CXX20 void generate(_ForwardIterator __first, _ForwardIterator __last, _Generator __gen) {
  std::__generate(__first, __last, __gen);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_GENERATE_H
