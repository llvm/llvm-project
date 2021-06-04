//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_PARTITION_H
#define _LIBCPP___ALGORITHM_PARTITION_H

#include <__config>
#include <__iterator/iterator_traits.h>
#include <type_traits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// partition

template <class _Predicate, class _ForwardIterator>
_LIBCPP_CONSTEXPR_AFTER_CXX17 _ForwardIterator
__partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred, forward_iterator_tag)
{
    while (true)
    {
        if (__first == __last)
            return __first;
        if (!__pred(*__first))
            break;
        ++__first;
    }
    for (_ForwardIterator __p = __first; ++__p != __last;)
    {
        if (__pred(*__p))
        {
            swap(*__first, *__p);
            ++__first;
        }
    }
    return __first;
}

template <class _Predicate, class _BidirectionalIterator>
_LIBCPP_CONSTEXPR_AFTER_CXX17 _BidirectionalIterator
__partition(_BidirectionalIterator __first, _BidirectionalIterator __last, _Predicate __pred,
            bidirectional_iterator_tag)
{
    while (true)
    {
        while (true)
        {
            if (__first == __last)
                return __first;
            if (!__pred(*__first))
                break;
            ++__first;
        }
        do
        {
            if (__first == --__last)
                return __first;
        } while (!__pred(*__last));
        swap(*__first, *__last);
        ++__first;
    }
}

template <class _ForwardIterator, class _Predicate>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
_ForwardIterator
partition(_ForwardIterator __first, _ForwardIterator __last, _Predicate __pred)
{
    return _VSTD::__partition<typename add_lvalue_reference<_Predicate>::type>
                            (__first, __last, __pred, typename iterator_traits<_ForwardIterator>::iterator_category());
}

// partition_copy

template <class _InputIterator, class _OutputIterator1,
          class _OutputIterator2, class _Predicate>
_LIBCPP_CONSTEXPR_AFTER_CXX17 pair<_OutputIterator1, _OutputIterator2>
partition_copy(_InputIterator __first, _InputIterator __last,
               _OutputIterator1 __out_true, _OutputIterator2 __out_false,
               _Predicate __pred)
{
    for (; __first != __last; ++__first)
    {
        if (__pred(*__first))
        {
            *__out_true = *__first;
            ++__out_true;
        }
        else
        {
            *__out_false = *__first;
            ++__out_false;
        }
    }
    return pair<_OutputIterator1, _OutputIterator2>(__out_true, __out_false);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_PARTITION_H
