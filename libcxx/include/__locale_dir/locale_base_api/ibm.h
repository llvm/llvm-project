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

namespace {

struct __setAndRestore {
  explicit __setAndRestore(locale_t locale) {
    if (locale == (locale_t)0) {
      __cloc   = newlocale(LC_ALL_MASK, "C", /* base */ (locale_t)0);
      __stored = uselocale(__cloc);
    } else {
      __stored = uselocale(locale);
    }
  }

  ~__setAndRestore() {
    uselocale(__stored);
    if (__cloc)
      freelocale(__cloc);
  }

private:
  locale_t __stored = (locale_t)0;
  locale_t __cloc   = (locale_t)0;
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
  __builtin_va_copy(ap_copy, ap);
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

namespace __ibm {
_LIBCPP_EXPORTED_FROM_ABI int isalnum_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isalpha_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isblank_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iscntrl_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isgraph_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int islower_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isprint_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int ispunct_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isspace_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isupper_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswalnum_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswalpha_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswblank_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswcntrl_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswdigit_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswgraph_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswlower_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswprint_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswpunct_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswspace_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswupper_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int iswxdigit_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int toupper_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int tolower_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI wint_t towupper_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI wint_t towlower_l(wint_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int strcoll_l(const char *, const char *, locale_t);
_LIBCPP_EXPORTED_FROM_ABI size_t strxfrm_l(char *, const char *, size_t, locale_t);
_LIBCPP_EXPORTED_FROM_ABI size_t strftime_l(char *, size_t , const char *, const struct tm *, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int wcscoll_l(const wchar_t *, const wchar_t *, locale_t);
_LIBCPP_EXPORTED_FROM_ABI size_t wcsxfrm_l(wchar_t *, const wchar_t *, size_t , locale_t);
}
#endif // _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_IBM_H
