//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_FILL_H
#define _LIBCPP___FWD_FILL_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _ForwardIterator,
          class _Sentinel,
          class _Tp,
          __enable_if_t<__has_forward_iterator_category<_ForwardIterator>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_ForwardIterator __first, _Sentinel __last, const _Tp& __value);

template <class _RandomAccessIterator,
          class _Tp,
          __enable_if_t<__has_random_access_iterator_category<_RandomAccessIterator>::value &&
                            !__is_segmented_iterator<_RandomAccessIterator>::value,
                        int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value);

template <class _SegmentedIterator,
          class _Tp,
          __enable_if_t<__is_segmented_iterator<_SegmentedIterator>::value, int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 void
__fill(_SegmentedIterator __first, _SegmentedIterator __last, const _Tp& __value);

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_FILL_H
