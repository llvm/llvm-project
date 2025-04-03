//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOR_EACH_N_SEGMENT_H
#define _LIBCPP___ALGORITHM_FOR_EACH_N_SEGMENT_H

#include <__config>
#include <__iterator/distance.h>
#include <__iterator/iterator_traits.h>
#include <__iterator/next.h>
#include <__iterator/segmented_iterator.h>
#include <__utility/convert_to_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// __for_each_n_segment optimizes linear iteration over segmented iterators. It processes a segmented
// input range defined by [__first, __first + __n), where __first is the starting segmented iterator
// and __n is the number of elements to process. The functor __func is applied to each segment using
// local iterator pairs for that segment. The return value of __func is ignored, and the function
// returns an iterator pointing to one past the last processed element in the input range.

template <class _SegmentedIterator, class _Size, class _Functor>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _SegmentedIterator
__for_each_n_segment(_SegmentedIterator __first, _Size __orig_n, _Functor __func) {
  static_assert(__is_segmented_iterator<_SegmentedIterator>::value &&
                    __has_random_access_iterator_category<
                        typename __segmented_iterator_traits<_SegmentedIterator>::__local_iterator>::value,
                "__for_each_n_segment only works with segmented iterators with random-access local iterators");
  if (__orig_n == 0)
    return __first;

  using _Traits = __segmented_iterator_traits<_SegmentedIterator>;
  typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
  _IntegralSize __n = __orig_n;
  auto __seg        = _Traits::__segment(__first);
  auto __sfirst     = _Traits::__begin(__seg);
  auto __slast      = _Traits::__end(__seg);
  auto __lfirst     = _Traits::__local(__first);
  auto __seg_size   = static_cast<_IntegralSize>(std::distance(__lfirst, __slast));

  // Single-segment case: input range fits within a single segment (may not align with segment boundaries)
  if (__n <= __seg_size) {
    auto __llast = std::next(__lfirst, __n);
    __func(__lfirst, __llast);
    return _Traits::__compose(__seg, __llast);
  }

  // Multi-segment case: input range spans multiple segments.
  // Process the first segment which might not start at the beginning of the segment
  __func(__lfirst, __slast);
  ++__seg;
  __n -= __seg_size;

  // Process the 2nd to last segments guaranteed to start at the beginning of each segment
  while (true) {
    __sfirst   = _Traits::__begin(__seg);
    __slast    = _Traits::__end(__seg);
    __seg_size = std::distance(__sfirst, __slast);

    // The last (potentially partial) segment
    if (__n <= __seg_size) {
      auto __llast = std::next(__sfirst, __n);
      __func(__sfirst, __llast);
      return _Traits::__compose(__seg, __llast);
    }

    // Middle whole segments that are completely in the range
    __func(__sfirst, __slast);
    ++__seg;
    __n -= __seg_size;
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOR_EACH_N_SEGMENT_H
