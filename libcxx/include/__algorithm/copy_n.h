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
#include <__algorithm/iterator_operations.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__type_traits/enable_if.h>
#include <__utility/convert_to_integral.h>
#include <__utility/move.h>
#include <__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _AlgPolicy,
          class _InIter,
          class _OutIter,
          __enable_if_t<__has_random_access_iterator_category<_InIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter, _OutIter>
__copy_n(_InIter __first, typename _IterOps<_AlgPolicy>::template __difference_type<_InIter> __n, _OutIter __result) {
  return std::__copy(__first, __first + __n, std::move(__result));
}

template <class _AlgPolicy,
          class _InIter,
          class _OutIter,
          __enable_if_t<!__has_random_access_iterator_category<_InIter>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 pair<_InIter, _OutIter>
__copy_n(_InIter __first, typename _IterOps<_AlgPolicy>::template __difference_type<_InIter> __n, _OutIter __result) {
  while (__n != 0) {
    *__result = *__first;
    ++__first;
    ++__result;
    --__n;
  }
  return std::make_pair(std::move(__first), std::move(__result));
}

// The InputIterator case is handled specially here because it's been written in a way to avoid incrementing __first
// if not absolutely required. This was done to allow its use with istream_iterator and we want to avoid breaking
// people, at least currently.
// See https://github.com/llvm/llvm-project/commit/99847d2bf132854fffa019bab19818768102ccad
template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          __enable_if_t<__has_exactly_input_iterator_category<_InputIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
copy_n(_InputIterator __first, _Size __n, _OutputIterator __result) {
  using _IntegralSize       = decltype(std::__convert_to_integral(__n));
  _IntegralSize __converted = __n;
  if (__converted > 0) {
    *__result = *__first;
    ++__result;
    for (--__converted; __converted > 0; --__converted) {
      ++__first;
      *__result = *__first;
      ++__result;
    }
  }
  return __result;
}

template <class _InputIterator,
          class _Size,
          class _OutputIterator,
          __enable_if_t<!__has_exactly_input_iterator_category<_InputIterator>::value, int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
copy_n(_InputIterator __first, _Size __n, _OutputIterator __result) {
  using _IntegralSize       = decltype(std::__convert_to_integral(__n));
  _IntegralSize __converted = __n;
  return std::__copy_n<_ClassicAlgPolicy>(__first, __iterator_difference_type<_InputIterator>(__converted), __result)
      .second;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_COPY_N_H
