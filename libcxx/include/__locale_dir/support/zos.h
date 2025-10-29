//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H

#include <__support/ibm/locale_mgmt_zos.h>
#include <__support/ibm/vasprintf.h>

#include "cstdlib"
#include <clocale> // std::lconv
#include <locale.h>
#include <stdarg.h>
#include <stdio.h>

#include <wctype.h>

_LIBCPP_BEGIN_NAMESPACE_STD

// These functions are exported within std namespace
// for compatibility with previous versions.
_LIBCPP_EXPORTED_FROM_ABI int isdigit_l(int, locale_t);
_LIBCPP_EXPORTED_FROM_ABI int isxdigit_l(int, locale_t);

namespace __locale {
struct __locale_guard {
  _LIBCPP_HIDE_FROM_ABI __locale_guard(locale_t& __loc) : __old_loc_(std::uselocale(__loc)) {}

  _LIBCPP_HIDE_FROM_ABI ~__locale_guard() {
    if (__old_loc_ != (locale_t)0)
      std::uselocale(__old_loc_);
  }

  locale_t __old_loc_;

  __locale_guard(__locale_guard const&)            = delete;
  __locale_guard& operator=(__locale_guard const&) = delete;
};

//
// Locale management
//
#define _LIBCPP_COLLATE_MASK LC_COLLATE_MASK
#define _LIBCPP_CTYPE_MASK LC_CTYPE_MASK
#define _LIBCPP_MONETARY_MASK LC_MONETARY_MASK
#define _LIBCPP_NUMERIC_MASK LC_NUMERIC_MASK
#define _LIBCPP_TIME_MASK LC_TIME_MASK
#define _LIBCPP_MESSAGES_MASK LC_MESSAGES_MASK
#define _LIBCPP_ALL_MASK LC_ALL_MASK
#define _LIBCPP_LC_ALL LC_ALL

#define _LIBCPP_CLOC std::__c_locale()
#ifndef _LIBCPP_LC_GLOBAL_LOCALE
#  define _LIBCPP_LC_GLOBAL_LOCALE ((locale_t) - 1)
#endif

using __locale_t _LIBCPP_NODEBUG = locale_t;

#if defined(_LIBCPP_BUILDING_LIBRARY)
using __lconv_t _LIBCPP_NODEBUG = std::lconv;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __locale, __locale_t __base) {
  return newlocale(__category_mask, __locale, __base);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI char* __setlocale(int __category, char const* __locale) {
  return ::setlocale(__category, __locale);
}

inline _LIBCPP_HIDE_FROM_ABI __lconv_t* __localeconv(__locale_t& __loc) {
  __locale_guard __current(__loc);
  return std::localeconv();
}
#endif // _LIBCPP_BUILDING_LIBRARY

// The following are not POSIX routines.  These are quick-and-dirty hacks
// to make things pretend to work

//
// Strtonum functions
//
inline _LIBCPP_HIDE_FROM_ABI float __strtof(const char* __nptr, char** __endptr, __locale_t __loc) {
  __locale_guard __newloc(__loc);
  return ::strtof(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI double __strtod(const char* __nptr, char** __endptr, __locale_t __loc) {
  __locale_guard __newloc(__loc);
  return ::strtod(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long double __strtold(const char* __nptr, char** __endptr, __locale_t __loc) {
  __locale_guard __newloc(__loc);
  return ::strtold(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long long __strtoll(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  __locale_guard __newloc(__loc);
  return ::strtoll(__nptr, __endptr, __base);
}

inline _LIBCPP_HIDE_FROM_ABI unsigned long long
__strtoull(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  __locale_guard __newloc(__loc);
  return ::strtoull(__nptr, __endptr, __base);
}

//
// Character manipulation functions
//
namespace __ibm {
_LIBCPP_HIDE_FROM_ABI int islower_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isupper_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswalpha_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswblank_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswcntrl_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswdigit_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswlower_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswprint_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswpunct_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswspace_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswupper_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswxdigit_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int toupper_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int tolower_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI wint_t towupper_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI wint_t towlower_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int strcoll_l(const char*, const char*, __locale_t);
_LIBCPP_HIDE_FROM_ABI size_t strxfrm_l(char*, const char*, size_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI size_t strftime_l(char*, size_t, const char*, const struct tm*, __locale_t);
_LIBCPP_HIDE_FROM_ABI int wcscoll_l(const wchar_t*, const wchar_t*, __locale_t);
_LIBCPP_HIDE_FROM_ABI size_t wcsxfrm_l(wchar_t*, const wchar_t*, size_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswctype_l(wint_t, wctype_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI size_t mbsnrtowcs(wchar_t*, const char**, size_t, size_t, mbstate_t*);
_LIBCPP_HIDE_FROM_ABI size_t wcsnrtombs(char*, const wchar_t**, size_t, size_t, mbstate_t*);

// These functions are not used internally by libcxx
// and are included for completness.
_LIBCPP_HIDE_FROM_ABI int isalnum_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isalpha_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isblank_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iscntrl_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isgraph_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isprint_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int ispunct_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int isspace_l(int, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswalnum_l(wint_t, __locale_t);
_LIBCPP_HIDE_FROM_ABI int iswgraph_l(wint_t, __locale_t);
} // namespace __ibm

using namespace __ibm;

#if defined(_LIBCPP_BUILDING_LIBRARY)
inline _LIBCPP_HIDE_FROM_ABI int __islower(int __c, __locale_t __loc) { return islower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __isupper(int __c, __locale_t __loc) { return isupper_l(__c, __loc); }
#endif

inline _LIBCPP_HIDE_FROM_ABI int __isdigit(int __c, __locale_t __loc) { return isdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __isxdigit(int __c, __locale_t __loc) { return isxdigit_l(__c, __loc); }

#if defined(_LIBCPP_BUILDING_LIBRARY)
inline _LIBCPP_HIDE_FROM_ABI int __toupper(int __c, __locale_t __loc) { return toupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __tolower(int __c, __locale_t __loc) { return tolower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __strcoll(const char* __s1, const char* __s2, __locale_t __loc) {
  return strcoll_l(__s1, __s2, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __strxfrm(char* __dest, const char* __src, size_t __n, __locale_t __loc) {
  return strxfrm_l(__dest, __src, __n, __loc);
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI int __iswctype(wint_t __c, wctype_t __type, __locale_t __loc) {
  return iswctype_l(__c, __type, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI int __iswspace(wint_t __c, __locale_t __loc) { return iswspace_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswprint(wint_t __c, __locale_t __loc) { return iswprint_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswcntrl(wint_t __c, __locale_t __loc) { return iswcntrl_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswupper(wint_t __c, __locale_t __loc) { return iswupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswlower(wint_t __c, __locale_t __loc) { return iswlower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswalpha(wint_t __c, __locale_t __loc) { return iswalpha_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswblank(wint_t __c, __locale_t __loc) { return iswblank_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswdigit(wint_t __c, __locale_t __loc) { return iswdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswpunct(wint_t __c, __locale_t __loc) { return iswpunct_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswxdigit(wint_t __c, __locale_t __loc) { return iswxdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI wint_t __towupper(wint_t __c, __locale_t __loc) { return towupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI wint_t __towlower(wint_t __c, __locale_t __loc) { return towlower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __wcscoll(const wchar_t* __ws1, const wchar_t* __ws2, __locale_t __loc) {
  return wcscoll_l(__ws1, __ws2, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __wcsxfrm(wchar_t* __dest, const wchar_t* __src, size_t __n, __locale_t __loc) {
  return wcsxfrm_l(__dest, __src, __n, __loc);
}
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS

inline _LIBCPP_HIDE_FROM_ABI
size_t __strftime(char* __s, size_t __max, const char* __format, const struct tm* __tm, __locale_t __loc) {
  return strftime_l(__s, __max, __format, __tm, __loc);
}

//
// Other functions
//
inline _LIBCPP_HIDE_FROM_ABI decltype(MB_CUR_MAX) __mb_len_max(__locale_t __loc) {
  __locale_guard __current(__loc);
  return MB_CUR_MAX;
}

#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI wint_t __btowc(int __c, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::btowc(__c);
}

inline _LIBCPP_HIDE_FROM_ABI int __wctob(wint_t __c, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wctob(__c);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__wcsnrtombs(char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return wcsnrtombs(__dest, __src, __nwc, __len, __ps); // non-standard
}

inline _LIBCPP_HIDE_FROM_ABI size_t __wcrtomb(char* __s, wchar_t __wc, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wcrtomb(__s, __wc, __ps);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbsnrtowcs(__dest, __src, __nms, __len, __ps); // non-standard
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbrtowc(wchar_t* __pwc, const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::mbrtowc(__pwc, __s, __n, __ps);
}

inline _LIBCPP_HIDE_FROM_ABI int __mbtowc(wchar_t* __pwc, const char* __pmb, size_t __max, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::mbtowc(__pwc, __pmb, __max);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __mbrlen(const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::mbrlen(__s, __n, __ps);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsrtowcs(wchar_t* __dest, const char** __src, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::mbsrtowcs(__dest, __src, __len, __ps);
}
#  endif // _LIBCPP_BUILDING_LIBRARY
#endif   // _LIBCPP_HAS_WIDE_CHARACTERS

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 4, 5) int __snprintf(
    char* __s, size_t __n, __locale_t __loc, const char* __format, ...) {
  va_list __va;
  va_start(__va, __format);
  __locale_guard __current(__loc);
  int __res = std::vsnprintf(__s, __n, __format, __va);
  va_end(__va);
  return __res;
}

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __asprintf(
    char** __s, __locale_t __loc, const char* __format, ...) {
  va_list __va;
  va_start(__va, __format);
  __locale_guard __current(__loc);
  int __res = std::__ibm::__vasprintf(__s, __format, __va); // non-standard
  va_end(__va);
  return __res;
}

_LIBCPP_HIDE_FROM_ABI inline _LIBCPP_ATTRIBUTE_FORMAT(__scanf__, 3, 4) int __sscanf(
    const char* __s, __locale_t __loc, const char* __format, ...) {
  va_list __va;
  va_start(__va, __format);
  __locale_guard __current(__loc);
  int __res = std::vsscanf(__s, __format, __va);
  va_end(__va);
  return __res;
}
} // namespace __locale
_LIBCPP_END_NAMESPACE_STD
#endif // _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H
