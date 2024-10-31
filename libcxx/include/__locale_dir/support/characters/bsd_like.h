//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_CHARACTERS_BSD_LIKE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_CHARACTERS_BSD_LIKE_H

#include <__config>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if _LIBCPP_HAS_WIDE_CHARACTERS
#  include <wchar.h>
#  include <wctype.h>
#endif

#include <xlocale.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

inline _LIBCPP_HIDE_FROM_ABI int __islower(int __c, __locale_t __loc) { return ::islower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __isupper(int __c, __locale_t __loc) { return ::isupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __isdigit(int __c, __locale_t __loc) { return ::isdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __isxdigit(int __c, __locale_t __loc) { return ::isxdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __toupper(int __c, __locale_t __loc) { return ::toupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __tolower(int __c, __locale_t __loc) { return ::tolower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __strcoll(const char* __s1, const char* __s2, __locale_t __loc) {
  return ::strcoll_l(__s1, __s2, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __strxfrm(char* __dest, const char* __src, size_t __n, __locale_t __loc) {
  return ::strxfrm_l(__dest, __src, __n, __loc);
}

#if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI int __iswspace(wint_t __c, __locale_t __loc) { return ::iswspace_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswprint(wint_t __c, __locale_t __loc) { return ::iswprint_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswcntrl(wint_t __c, __locale_t __loc) { return ::iswcntrl_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswupper(wint_t __c, __locale_t __loc) { return ::iswupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswlower(wint_t __c, __locale_t __loc) { return ::iswlower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswalpha(wint_t __c, __locale_t __loc) { return ::iswalpha_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswblank(wint_t __c, __locale_t __loc) { return ::iswblank_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswdigit(wint_t __c, __locale_t __loc) { return ::iswdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswpunct(wint_t __c, __locale_t __loc) { return ::iswpunct_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __iswxdigit(wint_t __c, __locale_t __loc) { return ::iswxdigit_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI wint_t __towupper(wint_t __c, __locale_t __loc) { return ::towupper_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI wint_t __towlower(wint_t __c, __locale_t __loc) { return ::towlower_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __wcscoll(const wchar_t* __ws1, const wchar_t* __ws2, __locale_t __loc) {
  return ::wcscoll_l(__ws1, __ws2, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __wcsxfrm(wchar_t* __dest, const wchar_t* __src, size_t __n, __locale_t __loc) {
  return ::wcsxfrm_l(__dest, __src, __n, __loc);
}
#endif // _LIBCPP_HAS_WIDE_CHARACTERS

inline _LIBCPP_HIDE_FROM_ABI size_t
__strftime(char* __s, size_t __max, const char* __format, const struct tm* __tm, __locale_t __loc) {
  return ::strftime_l(__s, __max, __format, __tm, __loc);
}

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_CHARACTERS_BSD_LIKE_H
