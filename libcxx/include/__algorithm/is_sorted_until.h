//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_IS_SORTED_UNTIL_H
#define _LIBCPP___ALGORITHM_IS_SORTED_UNTIL_H

#include <__algorithm/adjacent_find.h>
#include <__algorithm/comp.h>
#include <__algorithm/comp_ref_type.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Compare, class _ForwardIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
__is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  return std::adjacent_find(__first, __last, [&](decltype(*__first) __lhs, decltype(*__first) __rhs) {
    return __comp(__rhs, __lhs);
  });
}

template <class _ForwardIterator, class _Compare>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last, _Compare __comp) {
  return std::__is_sorted_until<__comp_ref_type<_Compare> >(__first, __last, __comp);
}

template <class _ForwardIterator>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _ForwardIterator
is_sorted_until(_ForwardIterator __first, _ForwardIterator __last) {
  return std::is_sorted_until(__first, __last, __less<>());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_IS_SORTED_UNTIL_H
