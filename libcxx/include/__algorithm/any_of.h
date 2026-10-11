// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_ANY_OF_H
#define _LIBCPP___ALGORITHM_ANY_OF_H

#include <__algorithm/find_if.h>
#include <__config>
#include <__functional/identity.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class _Proj, class _Pred>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
__any_of(_Iter __first, _Sent __last, _Pred& __pred, _Proj& __proj) {
  return std::__find_if(std::move(__first), __last, __pred, __proj) != __last;
}

template <class _InputIterator, class _Predicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool
any_of(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  __identity __proj;
  return std::__any_of(std::move(__first), std::move(__last), __pred, __proj);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_ANY_OF_H
