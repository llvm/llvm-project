//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_UNWRAP_RANGE_H
#define _LIBCPP___CXX03___ALGORITHM_UNWRAP_RANGE_H

#include <__cxx03/__algorithm/unwrap_iter.h>
#include <__cxx03/__config>
#include <__cxx03/__iterator/next.h>
#include <__cxx03/__utility/declval.h>
#include <__cxx03/__utility/move.h>
#include <__cxx03/__utility/pair.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

// __unwrap_range and __rewrap_range are used to unwrap ranges which may have different iterator and sentinel types.
// __unwrap_iter and __rewrap_iter don't work for this, because they assume that the iterator and sentinel have
// the same type. __unwrap_range tries to get two iterators and then forward to __unwrap_iter.

template <class _Iter, class _Unwrapped = decltype(std::__unwrap_iter(std::declval<_Iter>()))>
_LIBCPP_HIDE_FROM_ABI pair<_Unwrapped, _Unwrapped> __unwrap_range(_Iter __first, _Iter __last) {
  return std::make_pair(std::__unwrap_iter(std::move(__first)), std::__unwrap_iter(std::move(__last)));
}

template <class _Iter, class _Unwrapped>
_LIBCPP_HIDE_FROM_ABI _Iter __rewrap_range(_Iter __orig_iter, _Unwrapped __iter) {
  return std::__rewrap_iter(std::move(__orig_iter), std::move(__iter));
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_UNWRAP_RANGE_H
