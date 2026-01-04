//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FILL_N_H
#define _LIBCPP___ALGORITHM_FILL_N_H

#include <__algorithm/for_each_n_segment.h>
#include <__algorithm/specialized_algorithms.h>
#include <__config>
#include <__iterator/iterator_traits.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/enable_if.h>
#include <__utility/convert_to_integral.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// fill_n isn't specialized for std::memset, because the compiler already optimizes the loop to a call to std::memset.

template <
    class _OutputIterator,
    class _Size,
    class _Tp,
    __enable_if_t<!__specialized_algorithm<_Algorithm::__fill_n, __single_iterator<_OutputIterator> >::__has_algorithm,
                  int> = 0>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
__fill_n(_OutputIterator __first, _Size __n, const _Tp& __value) {
#ifndef _LIBCPP_CXX03_LANG
  if constexpr (__is_segmented_iterator_v<_OutputIterator>) {
    using __local_iterator = typename __segmented_iterator_traits<_OutputIterator>::__local_iterator;
    if constexpr (__has_random_access_iterator_category<__local_iterator>::value) {
      return std::__for_each_n_segment(__first, __n, [&](__local_iterator __lfirst, __local_iterator __llast) {
        std::__fill_n(__lfirst, __llast - __lfirst, __value);
      });
    }
  }
#endif
  for (; __n > 0; ++__first, (void)--__n)
    *__first = __value;
  return __first;
}

template <class _OutIter,
          class _Size,
          class _Tp,
          __enable_if_t<__specialized_algorithm<_Algorithm::__fill_n, __single_iterator<_OutIter> >::__has_algorithm,
                        int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutIter __fill_n(_OutIter __first, _Size __n, const _Tp& __value) {
  return __specialized_algorithm<_Algorithm::__fill_n, __single_iterator<_OutIter> >()(
      std::move(__first), __n, __value);
}

template <class _OutputIterator, class _Size, class _Tp>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _OutputIterator
fill_n(_OutputIterator __first, _Size __n, const _Tp& __value) {
  return std::__fill_n(__first, std::__convert_to_integral(__n), __value);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_FILL_N_H
