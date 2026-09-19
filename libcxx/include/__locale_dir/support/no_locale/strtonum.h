//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_NO_LOCALE_STRTONUM_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_NO_LOCALE_STRTONUM_H

#include <__config>
#include <cstdlib>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

//
// Strtonum functions
//
template <class _FloatT>
_LIBCPP_HIDE_FROM_ABI _FloatT __str_to_float_c_locale(const char* __nptr, char** __endptr, __locale_t);

template <>
inline _LIBCPP_HIDE_FROM_ABI float __str_to_float_c_locale<float>(const char* __nptr, char** __endptr, __locale_t) {
  return std::strtof(__nptr, __endptr);
}

template <>
inline _LIBCPP_HIDE_FROM_ABI double __str_to_float_c_locale<double>(const char* __nptr, char** __endptr, __locale_t) {
  return std::strtod(__nptr, __endptr);
}

template <>
inline _LIBCPP_HIDE_FROM_ABI long double
__str_to_float_c_locale<long double>(const char* __nptr, char** __endptr, __locale_t) {
  return std::strtold(__nptr, __endptr);
}

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_NO_LOCALE_STRTONUM_H
