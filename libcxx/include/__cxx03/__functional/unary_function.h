//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CXX03___FUNCTIONAL_UNARY_FUNCTION_H
#define _LIBCPP___CXX03___FUNCTIONAL_UNARY_FUNCTION_H

#include <__cxx03/__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Arg, class _Result>
struct _LIBCPP_TEMPLATE_VIS unary_function {
  typedef _Arg argument_type;
  typedef _Result result_type;
};

template <class _Arg, class _Result>
struct __unary_function_keep_layout_base {
  using argument_type = _Arg;
  using result_type   = _Result;
};

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wdeprecated-declarations")
template <class _Arg, class _Result>
using __unary_function = unary_function<_Arg, _Result>;
_LIBCPP_DIAGNOSTIC_POP

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CXX03___FUNCTIONAL_UNARY_FUNCTION_H
