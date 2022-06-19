//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_POP_HEAP_H
#define _LIBCPP___ALGORITHM_POP_HEAP_H

#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__algorithm/push_heap.h>
#include <__algorithm/sift_down.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Compare, class _RandomAccessIterator>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
__pop_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp,
           typename iterator_traits<_RandomAccessIterator>::difference_type __len)
{
    using value_type = typename iterator_traits<_RandomAccessIterator>::value_type;

    if (__len > 1)
    {
        value_type __top = std::move(*__first);  // create a hole at __first
        _RandomAccessIterator __hole = std::__floyd_sift_down<_Compare>(__first, __comp, __len);
        --__last;
        if (__hole == __last) {
            *__hole = std::move(__top);
        } else {
            *__hole = std::move(*__last);
            ++__hole;
            *__last = std::move(__top);
            std::__sift_up<_Compare>(__first, __hole, __comp, __hole - __first);
        }
    }
}

template <class _RandomAccessIterator, class _Compare>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
pop_heap(_RandomAccessIterator __first, _RandomAccessIterator __last, _Compare __comp)
{
    typedef typename __comp_ref_type<_Compare>::type _Comp_ref;
    _VSTD::__pop_heap<_Comp_ref>(__first, __last, __comp, __last - __first);
}

template <class _RandomAccessIterator>
inline _LIBCPP_INLINE_VISIBILITY _LIBCPP_CONSTEXPR_AFTER_CXX17
void
pop_heap(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
    _VSTD::pop_heap(__first, __last, __less<typename iterator_traits<_RandomAccessIterator>::value_type>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_POP_HEAP_H
