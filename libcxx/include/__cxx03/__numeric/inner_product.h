// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___NUMERIC_INNER_PRODUCT_H
#define _LIBCPP___CXX03___NUMERIC_INNER_PRODUCT_H

#include <__cxx03/__config>
#include <__cxx03/__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__cxx03/__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _InputIterator1, class _InputIterator2, class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp
inner_product(_InputIterator1 __first1, _InputIterator1 __last1, _InputIterator2 __first2, _Tp __init) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    __init = __init + *__first1 * *__first2;
  return __init;
}

template <class _InputIterator1, class _InputIterator2, class _Tp, class _BinaryOperation1, class _BinaryOperation2>
_LIBCPP_HIDE_FROM_ABI _Tp inner_product(
    _InputIterator1 __first1,
    _InputIterator1 __last1,
    _InputIterator2 __first2,
    _Tp __init,
    _BinaryOperation1 __binary_op1,
    _BinaryOperation2 __binary_op2) {
  for (; __first1 != __last1; ++__first1, (void)++__first2)
    __init = __binary_op1(__init, __binary_op2(*__first1, *__first2));
  return __init;
}

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CXX03___NUMERIC_INNER_PRODUCT_H
