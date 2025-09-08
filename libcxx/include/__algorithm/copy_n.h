//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_COPY_N_H
#define _LIBCPP___ALGORITHM_COPY_N_H

#include <__algorithm/copy.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/enable_if.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InIter,
          class _DiffType,
          class _OutIter,
          __enable_if_t<__has_random_access_iterator_category<_InIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter, _OutIter>
__copy_n(_InIter __first, _DiffType __n, _OutIter __result) {
  return std::__copy(__first, __first + __n, std::move(__result));
}

template <class _InIter,
          class _DiffType,
          class _OutIter,
          __enable_if_t<!__has_random_access_iterator_category<_InIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter, _OutIter>
__copy_n(_InIter __first, _DiffType __n, _OutIter __result) {
  while (__n != 0) {
    *__result = *__first;
    ++__first;
    ++__result;
    --__n;
  }
  return std::make_pair(std::move(__first), std::move(__result));
}

template <class _InputIterator, class _Size, class _OutputIterator>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
copy_n(_InputIterator __first, _Size __n, _OutputIterator __result) {
  using __diff_t = __iter_diff_t<_InputIterator>;
  return std::__copy_n(__first, __diff_t(__n), __result).second;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_COPY_N_H
