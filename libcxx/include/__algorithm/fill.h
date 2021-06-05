//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FILL_H
#define _LIBCPP___ALGORITHM_FILL_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// fill_n

template <class _OutputIterator, class _Size, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
_OutputIterator
__fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
    for (; __n > 0; ++__first, (void) --__n)
        *__first = __value_;
    return __first;
}

template <class _OutputIterator, class _Size, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
_OutputIterator
fill_n(_OutputIterator __first, _Size __n, const _Tp& __value_)
{
   return _VSTD::__fill_n(__first, _VSTD::__convert_to_integral(__n), __value_);
}

// fill

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
__fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value_, forward_iterator_tag)
{
    for (; __first != __last; ++__first)
        *__first = __value_;
}

template <class _RandomAccessIterator, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
__fill(_RandomAccessIterator __first, _RandomAccessIterator __last, const _Tp& __value_, random_access_iterator_tag)
{
    _VSTD::fill_n(__first, __last - __first, __value_);
}

template <class _ForwardIterator, class _Tp>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
fill(_ForwardIterator __first, _ForwardIterator __last, const _Tp& __value_)
{
    _VSTD::__fill(__first, __last, __value_, typename iterator_traits<_ForwardIterator>::iterator_category());
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FILL_H
