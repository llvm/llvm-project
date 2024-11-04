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
#include <__algorithm/for_each.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// fill isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template < class _ForwardIterator,
           class _Tp,
           __enable_if_t<!__has_random_access_iterator_category<_ForwardIterator>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  for (; __first != __last; ++__first)
    *__first = __value;
}

template <class _RandomAccessIterator,
          class _Tp,
          __enable_if_t<__has_random_access_iterator_category<_RandomAccessIterator>::value &&
                            !__is_segmented_iterator<_RandomAccessIterator>::value,
                        int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value) {
  std::fill_n(__first, __last - __first, __value);
}

template <class _Tp>
struct __fill_segment {
  const _Tp& __value_;

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR __fill_segment(const _Tp& __value) : __value_(__value) {}

  _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 void operator()(_Tp& __val) const { __val = __value_; }
};

template <class _SegmentedIterator,
          class _Tp,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_SegmentedIterator __first, _SegmentedIterator __last, const _Tp& __value) {
  std::for_each(__first, __last, __fill_segment<_Tp>(__value));
}

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value) {
  std::__fill(__first, __last, __value);
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FILL_H
