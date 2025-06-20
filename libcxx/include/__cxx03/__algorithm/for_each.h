// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___ALGORITHM_FOR_EACH_H
#define _LIBCPP___CXX03___ALGORITHM_FOR_EACH_H

#include <__cxx03/__algorithm/for_each_segment.h>
#include <__cxx03/__config>
#include <__cxx03/__iterator/segmented_iterator.h>
#include <__cxx03/__type_traits/enable_if.h>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator, class _Function>
_LIBCPP_HIDE_FROM_ABI _Function for_each(_InputIterator __first, _InputIterator __last, _Function __f) {
  for (; __first != __last; ++__first)
    __f(*__first);
  return __f;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___ALGORITHM_FOR_EACH_H
