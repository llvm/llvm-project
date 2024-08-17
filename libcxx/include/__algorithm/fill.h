//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FILL_H
#define _LIBCPP___ALGORITHM_FILL_H

#include <__algorithm/fill_n.h>
#include <__config>
#include <__iterator/iterator_traits.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// fill isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template <
    class _ForwardIterator,
    class _Tp,
    __enable_if_t<
        is_same<typename iterator_traits<_ForwardIterator>::iterator_category, forward_iterator_tag>::value ||
            is_same<typename iterator_traits<_ForwardIterator>::iterator_category, bidirectional_iterator_tag>::value,
        int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  for (; __first != __last; ++__first)
    *__first = __value;
}

template <class _RandomAccessIterator,
          class _Tp,
          __enable_if_t<(is_same<typename iterator_traits<_RandomAccessIterator>::iterator_category,
                                 random_access_iterator_tag>::value ||
                         is_same<typename iterator_traits<_RandomAccessIterator>::iterator_category,
                                 contiguous_iterator_tag>::value) &&
                            !__is_segmented_iterator<_RandomAccessIterator>::value,
                        int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value) {
  std::fill_n(__first, __last - __first, __value);
}

template <class _SegmentedIterator,
          class _Tp,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_SegmentedIterator __first, _SegmentedIterator __last, const _Tp& __value) {
  using _Traits = __segmented_iterator_traits<_SegmentedIterator>;

  auto __sfirst = _Traits::__segment(__first);
  auto __slast  = _Traits::__segment(__last);

  // We are in a single segment, so we might not be at the beginning or end
  if (__sfirst == __slast) {
    __fill(_Traits::__local(__first), _Traits::__local(__last), __value);
    return;
  }

  // We have more than one segment. Iterate over the first segment, since we might not start at the beginning
  __fill(_Traits::__local(__first), _Traits::__end(__sfirst), __value);
  ++__sfirst;
  // iterate over the segments which are guaranteed to be completely in the range
  while (__sfirst != __slast) {
    __fill(_Traits::__begin(__sfirst), _Traits::__end(__sfirst), __value);
    ++__sfirst;
  }
  // iterate over the last segment
  __fill(_Traits::__begin(__sfirst), _Traits::__local(__last), __value);
}

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  std::__fill(__first, __last, __value);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FILL_H