// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_FIND_IF_H
#define _LIBCPP___ALGORITHM_FIND_IF_H

#include <__config>
#include <__functional/identity.h>
#include <__memory/valid_range.h>
#include <__type_traits/invoke.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class _Pred, class _Proj>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX17 _Iter
__find_if(_Iter __first, _Sent __last, _Pred&& __pred, _Proj&& __proj) {
  std::__assume_valid_range(__first, __last);

  for (; __first != __last; ++__first)
    if (std::__invoke(__pred, std::__invoke(__proj, *__first)))
      break;
  return __first;
}

template <class _InputIterator, class _Predicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 _InputIterator
find_if(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  return std::__find_if(__first, __last, __pred, __identity());
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___ALGORITHM_FIND_IF_H
