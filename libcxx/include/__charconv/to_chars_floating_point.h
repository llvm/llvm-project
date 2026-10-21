// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___CHARCONV_TO_CHARS_FLOATING_POINT_H
#define _LIBCPP___CHARCONV_TO_CHARS_FLOATING_POINT_H

#include <__charconv/chars_format.h>
#include <__charconv/to_chars_result.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 17

#  if _LIBCPP_AVAILABILITY_HAS_TO_CHARS_FLOATING_POINT

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_EXPORTED_FROM_ABI to_chars_result to_chars(char* __first, char* __last, float __value);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result to_chars(char* __first, char* __last, double __value);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result to_chars(char* __first, char* __last, long double __value);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result to_chars(char* __first, char* __last, float __value, chars_format __fmt);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result to_chars(char* __first, char* __last, double __value, chars_format __fmt);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, long double __value, chars_format __fmt);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, float __value, chars_format __fmt, int __precision);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, double __value, chars_format __fmt, int __precision);

_LIBCPP_EXPORTED_FROM_ABI to_chars_result
to_chars(char* __first, char* __last, long double __value, chars_format __fmt, int __precision);

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

#  else

// When the floating-point overloads of std::to_chars are not provided by the built library on the deployment
// target, we define them as inline functions that forward to a function that doesn't exist anywhere. Using
// them then results in a link-time error that mentions the cause of the problem. This is used instead of
// availability attributes because Clang diagnoses them too eagerly (http://llvm.org/PR61563). With this approach,
// there is only an error when the code is actually emitted.
to_chars_result __to_chars_floating_point_is_not_available_on_this_deployment_target();

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, float) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, double) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, long double) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, float, chars_format) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, double, chars_format) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, long double, chars_format) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, float, chars_format, int) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, double, chars_format, int) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

_LIBCPP_HIDE_FROM_ABI inline to_chars_result to_chars(char*, char*, long double, chars_format, int) {
  return std::__to_chars_floating_point_is_not_available_on_this_deployment_target();
}

#  endif // _LIBCPP_AVAILABILITY_HAS_TO_CHARS_FLOATING_POINT

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___CHARCONV_TO_CHARS_FLOATING_POINT_H
