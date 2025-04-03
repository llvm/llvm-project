// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOR_EACH_N_H
#define _LIBCPP___ALGORITHM_FOR_EACH_N_H

#include <__algorithm/for_each.h>
#include <__algorithm/for_each_n_segment.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/invoke.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>
#include <__utility/convert_to_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator,
          class _Size,
          class _Function,
          class _Proj,
          __enable_if_t<!__has_random_access_iterator_category<_InputIterator>::value &&
                            (!__is_segmented_iterator<_InputIterator>::value
                             //   || !__has_random_access_iterator_category<
                             //      typename __segmented_iterator_traits<_InputIterator>::__local_iterator>::value
                             ), // TODO: __segmented_iterator_traits<_InputIterator> results in template instantiation
                                // during SFINAE, which is a hard error to be fixed. Once fixed, we should uncomment.
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
__for_each_n(_InputIterator __first, _Size __orig_n, _Function& __f, _Proj& __proj) {
  typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
  _IntegralSize __n = __orig_n;
  while (__n > 0) {
    std::invoke(__f, std::invoke(__proj, *__first));
    ++__first;
    --__n;
  }
  return __first;
}

template <class _RandIter,
          class _Size,
          class _Function,
          class _Proj,
          __enable_if_t<__has_random_access_iterator_category<_RandIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _RandIter
__for_each_n(_RandIter __first, _Size __orig_n, _Function& __f, _Proj& __proj) {
  typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
  _IntegralSize __n = __orig_n;
  return std::__for_each(__first, __first + __n, __f, __proj);
}

template <class _SegmentedIterator,
          class _Size,
          class _Function,
          class _Proj,
          __enable_if_t<!__has_random_access_iterator_category<_SegmentedIterator>::value &&
                            __is_segmented_iterator<_SegmentedIterator>::value &&
                            __has_random_access_iterator_category<
                                typename __segmented_iterator_traits<_SegmentedIterator>::__local_iterator>::value,
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _SegmentedIterator
__for_each_n(_SegmentedIterator __first, _Size __orig_n, _Function& __f, _Proj& __proj) {
  return std::__for_each_n_segment(
      __first, __orig_n, std::__segment_processor<_SegmentedIterator, _Function, _Proj>(__f, __proj));
}

#if _LIBCPP_STD_VER >= 17

template <class _InputIterator, class _Size, class _Function>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
for_each_n(_InputIterator __first, _Size __orig_n, _Function __f) {
  __identity __proj;
  return std::__for_each_n(__first, __orig_n, __f, __proj);
}

#endif

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOR_EACH_N_H
