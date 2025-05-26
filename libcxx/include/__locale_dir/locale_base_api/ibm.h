// -*- C++ -*-
//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_IBM_H
#define _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_IBM_H

#if defined(__MVS__)
#  include <__support/ibm/locale_mgmt_zos.h>
#endif // defined(__MVS__)

#include <locale.h>
#include <stdarg.h>
#include <stdio.h>

#include "cstdlib"

#if defined(__MVS__)
#  include <wctype.h>
// POSIX routines
#  include <__support/xlocale/__posix_l_fallback.h>
#endif // defined(__MVS__)

_LIBCPP_BEGIN_NAMESPACE_STD
#ifdef __MVS__
#  define _LIBCPP_CLOC std::__c_locale()
#  ifndef _LIBCPP_LC_GLOBAL_LOCALE
#    define _LIBCPP_LC_GLOBAL_LOCALE ((locale_t) -1)
#  endif
#else
extern locale_t __cloc();
#  define _LIBCPP_CLOC std::__cloc()
#endif
_LIBCPP_END_NAMESPACE_STD

#ifdef __MVS__
_LIBCPP_BEGIN_NAMESPACE_STD
#endif

namespace {

struct __setAndRestore {
  explicit __setAndRestore(locale_t __l) {
    if (__l == (locale_t)0) {
      __stored = std::uselocale(_LIBCPP_CLOC);
    } else {
      __stored = std::uselocale(__l);
    }
  }

  ~__setAndRestore() { std::uselocale(__stored); }

private:
  locale_t __stored = (locale_t)0;
};

} // namespace

// The following are not POSIX routines.  These are quick-and-dirty hacks
// to make things pretend to work
inline _LIBCPP_HIDE_FROM_ABI long long strtoll_l(const char* __nptr, char** __endptr, int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return ::strtoll(__nptr, __endptr, __base);
}

inline _LIBCPP_HIDE_FROM_ABI double strtod_l(const char* __nptr, char** __endptr, locale_t locale) {
  __setAndRestore __newloc(locale);
  return ::strtod(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI float strtof_l(const char* __nptr, char** __endptr, locale_t locale) {
  __setAndRestore __newloc(locale);
  return ::strtof(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long double strtold_l(const char* __nptr, char** __endptr, locale_t locale) {
  __setAndRestore __newloc(locale);
  return ::strtold(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI unsigned long long
strtoull_l(const char* __nptr, char** __endptr, int __base, locale_t locale) {
  __setAndRestore __newloc(locale);
  return ::strtoull(__nptr, __endptr, __base);
}

inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_ATTRIBUTE_FORMAT(__printf__, 2, 0) int vasprintf(char** strp, const char* fmt, va_list ap) {
  const size_t buff_size = 256;
  if ((*strp = (char*)malloc(buff_size)) == nullptr) {
    return -1;
  }

  va_list ap_copy;
  // va_copy may not be provided by the C library in C++03 mode.
#if defined(_LIBCPP_CXX03_LANG) && __has_builtin(__builtin_va_copy)
#  if defined(__MVS__) && !defined(_VARARG_EXT_)
  __builtin_zos_va_copy(ap_copy, ap);
#  else
  __builtin_va_copy(ap_copy, ap);
#  endif
#else
  va_copy(ap_copy, ap);
#endif
  int str_size = vsnprintf(*strp, buff_size, fmt, ap_copy);
  va_end(ap_copy);

  if ((size_t)str_size >= buff_size) {
    if ((*strp = (char*)realloc(*strp, str_size + 1)) == nullptr) {
      return -1;
    }
    str_size = vsnprintf(*strp, str_size + 1, fmt, ap);
  }
  return str_size;
}

#ifdef __MVS__
_LIBCPP_END_NAMESPACE_STD
#endif

#endif // _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_IBM_H
