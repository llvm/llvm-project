// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHARCONV_FROM_CHARS_FLOATING_POINT_H
#define _LIBCPP___CHARCONV_FROM_CHARS_FLOATING_POINT_H

#include <__assert>
#include <__charconv/chars_format.h>
#include <__charconv/from_chars_result.h>
#include <__charconv/traits.h>
#include <__config>
#include <__system_error/errc.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_floating_point.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 17

_LIBCPP_AVAILABILITY_FROM_CHARS_FLOATING_POINT _LIBCPP_EXPORTED_FROM_ABI from_chars_result
__from_chars_floating_point(const char* __first, const char* __last, float& __value, chars_format __fmt);

_LIBCPP_AVAILABILITY_FROM_CHARS_FLOATING_POINT _LIBCPP_EXPORTED_FROM_ABI from_chars_result
__from_chars_floating_point(const char* __first, const char* __last, double& __value, chars_format __fmt);

_LIBCPP_HIDE_FROM_ABI inline from_chars_result
from_chars(const char* __first, const char* __last, float& __value, chars_format __fmt = chars_format::general) {
  return std::__from_chars_floating_point(__first, __last, __value, __fmt);
}

_LIBCPP_HIDE_FROM_ABI inline from_chars_result
from_chars(const char* __first, const char* __last, double& __value, chars_format __fmt = chars_format::general) {
  return std::__from_chars_floating_point(__first, __last, __value, __fmt);
}

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___CHARCONV_FROM_CHARS_FLOATING_POINT_H
