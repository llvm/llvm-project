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
#include <__iterator/next.h>
#include <__iterator/segmented_iterator.h>
#include <__utility/convert_to_integral.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// __for_each_n_segment is a utility function for optimizing iterating over segmented iterators linearly.
// __first and __orig_n are represent the begining and size of a segmented range. __func is expected to
// take a range of local iterators. Anything that is returned from __func is ignored.

template <class _SegmentedIterator, class _Size, class _Functor>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 _SegmentedIterator
__for_each_n_segment(_SegmentedIterator __first, _Size __orig_n, _Functor __func) {
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

  // We have only one single segment, which might not start or end at the boundaries of the segment
  if (__n <= __seg_size) {
    auto __llast = std::next(__lfirst, __n);
    __func(__lfirst, __llast);
    return _Traits::__compose(__seg, __llast);
  }

  // We have more than one segment. Iterate over the first segment which might not start at the beginning
  __func(__lfirst, std::next(__lfirst, __seg_size));
  ++__seg;
  __n -= __seg_size;

  // Iterate over the 2nd to last segments which are guaranteed to start at the beginning of each segment
  while (true) {
    __sfirst   = _Traits::__begin(__seg);
    __slast    = _Traits::__end(__seg);
    __seg_size = std::distance(__sfirst, __slast);

    // We are in the last segment
    if (__n <= __seg_size) {
      auto __llast = std::next(__sfirst, __n);
      __func(__sfirst, __llast);
      return _Traits::__compose(__seg, __llast);
    }

    // We are in middle segments that are completely in the range
    __func(__sfirst, __slast);
    ++__seg;
    __n -= __seg_size;
  }
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOR_EACH_N_SEGMENT_H
