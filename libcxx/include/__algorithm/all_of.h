// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_ALL_OF_H
#define _LIBCPP___ALGORITHM_ALL_OF_H

#include <__algorithm/any_of.h>
#include <__config>
#include <__functional/identity.h>
#include <__type_traits/invoke.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Iter, class _Sent, class _Proj, class _Pred>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 bool
__all_of(_Iter __first, _Sent __last, _Pred& __pred, _Proj& __proj) {
  using _Ref          = decltype(std::__invoke(__proj, *__first));
  auto __negated_pred = [&__pred](_Ref __arg) -> bool { return !std::__invoke(__pred, std::forward<_Ref>(__arg)); };
  return !std::__any_of(std::move(__first), std::move(__last), __negated_pred, __proj);
}

template <class _InputIterator, class _Predicate>
[[__nodiscard__]] inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX20 bool
all_of(_InputIterator __first, _InputIterator __last, _Predicate __pred) {
  __identity __proj;
  return std::__all_of(__first, __last, __pred, __proj);
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___ALGORITHM_ALL_OF_H
