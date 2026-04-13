//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_SWAP_RANGES_H
#define _LIBCPP___ALGORITHM_SWAP_RANGES_H

#include <__algorithm/iterator_operations.h>
#include <__algorithm/specialized_algorithms.h>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <
    class _AlgPolicy,
    class _Iter1,
    class _Sent1,
    class _Iter2,
    class _SpecialAlg =
        __specialized_algorithm<_Algorithm::__swap_ranges, __iterator_pair<_Iter1, _Sent1>, __single_iterator<_Iter2> >,
    __enable_if_t<_SpecialAlg::__has_algorithm, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_Iter1, _Iter2>
__swap_ranges(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2) {
  return _SpecialAlg()(std::move(__first1), std::move(__last1), std::move(__first2));
}

template <
    class _AlgPolicy,
    class _Iter1,
    class _Sent1,
    class _Iter2,
    class _SpecialAlg =
        __specialized_algorithm<_Algorithm::__swap_ranges, __iterator_pair<_Iter1, _Sent1>, __single_iterator<_Iter2> >,
    __enable_if_t<!_SpecialAlg::__has_algorithm, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_Iter1, _Iter2>
__swap_ranges(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2) {
  while (__first1 != __last1) {
    _IterOps<_AlgPolicy>::iter_swap(__first1, __first2);
    ++__first1;
    ++__first2;
  }

  return pair<_Iter1, _Iter2>(std::move(__first1), std::move(__first2));
}

template <class _ForwardIterator1, class _ForwardIterator2>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator2
swap_ranges(_ForwardIterator1 __first1, _ForwardIterator1 __last1, _ForwardIterator2 __first2) {
  return std::__swap_ranges<_ClassicAlgPolicy>(std::move(__first1), std::move(__last1), std::move(__first2)).second;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_SWAP_RANGES_H
