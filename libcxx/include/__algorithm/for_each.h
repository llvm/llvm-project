// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FOR_EACH_H
#define _LIBCPP___ALGORITHM_FOR_EACH_H

#include <__algorithm/for_each_segment.h>
#include <__config>
#include <__functional/identity.h>
#include <__iterator/segmented_iterator.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_same.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Sent, class _Func, class _Proj>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
__for_each(_InputIterator __first, _Sent __last, _Func& __func, _Proj& __proj) {
#ifndef _LIBCPP_CXX03_LANG
  if constexpr (is_same<_InputIterator, _Sent>::value && __is_segmented_iterator_v<_InputIterator>) {
    using __local_iterator_t = typename __segmented_iterator_traits<_InputIterator>::__local_iterator;
    std::__for_each_segment(__first, __last, [&](__local_iterator_t __lfirst, __local_iterator_t __llast) {
      std::__for_each(__lfirst, __llast, __func, __proj);
    });
    return __last;
  }
#endif
  for (; __first != __last; ++__first)
    std::__invoke(__func, std::__invoke(__proj, *__first));
  return __first;
}

template <class _InputIterator, class _Func>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _Func
for_each(_InputIterator __first, _InputIterator __last, _Func __f) {
  __identity __proj;
  std::__for_each(__first, __last, __f, __proj);
  return __f;
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FOR_EACH_H
