// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The BSDs have lots of *_l functions.  We don't want to define those symbols
// on other platforms though, for fear of conflicts with user code.  So here,
// we will define the mapping from an internal macro to the real BSD symbol.
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_BSD_LOCALE_DEFAULTS_H
#define _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_BSD_LOCALE_DEFAULTS_H

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#if _LIBCPP_HAS_WIDE_CHARACTERS
#  include <wchar.h>
#endif

// <xlocale.h> must come after the includes above since the functions it includes depend on
// what headers have been included up to that point.
#if defined(__APPLE__) || defined(__FreeBSD__)
#  include <xlocale.h>
#endif

#include <__config>
#include <__cstddef/size_t.h>
#include <__std_mbstate_t.h>
#include <__utility/forward.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

inline _LIBCPP_HIDE_FROM_ABI decltype(MB_CUR_MAX) __libcpp_mb_cur_max_l(locale_t __loc) { return MB_CUR_MAX_L(__loc); }

#if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI wint_t __libcpp_btowc_l(int __c, locale_t __loc) { return ::btowc_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __libcpp_wctob_l(wint_t __c, locale_t __loc) { return ::wctob_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI size_t __libcpp_wcsnrtombs_l(
    char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, locale_t __loc) {
  return ::wcsnrtombs_l(__dest, __src, __nwc, __len, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __libcpp_wcrtomb_l(char* __s, wchar_t __wc, mbstate_t* __ps, locale_t __loc) {
  return ::wcrtomb_l(__s, __wc, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __libcpp_mbsnrtowcs_l(
    wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, locale_t __loc) {
  return ::mbsnrtowcs_l(__dest, __src, __nms, __len, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__libcpp_mbrtowc_l(wchar_t* __pwc, const char* __s, size_t __n, mbstate_t* __ps, locale_t __loc) {
  return ::mbrtowc_l(__pwc, __s, __n, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI int __libcpp_mbtowc_l(wchar_t* __pwc, const char* __pmb, size_t __max, locale_t __loc) {
  return ::mbtowc_l(__pwc, __pmb, __max, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __libcpp_mbrlen_l(const char* __s, size_t __n, mbstate_t* __ps, locale_t __loc) {
  return ::mbrlen_l(__s, __n, __ps, __loc);
}
#endif // _LIBCPP_HAS_WIDE_CHARACTERS

inline _LIBCPP_HIDE_FROM_ABI lconv* __libcpp_localeconv_l(locale_t& __loc) { return ::localeconv_l(__loc); }

#if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI size_t
__libcpp_mbsrtowcs_l(wchar_t* __dest, const char** __src, size_t __len, mbstate_t* __ps, locale_t __loc) {
  return ::mbsrtowcs_l(__dest, __src, __len, __ps, __loc);
}
#endif

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wgcc-compat")
_LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wformat-nonliteral") // GCC doesn't support [[gnu::format]] on variadic templates
#ifdef _LIBCPP_COMPILER_CLANG_BASED
#  define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(...) _LIBCPP_ATTRIBUTE_FORMAT(__VA_ARGS__)
#else
#  define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT
#endif

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 4, 5) int __libcpp_snprintf_l(
    char* __s, size_t __n, locale_t __loc, const char* __format, _Args&&... __args) {
  return ::snprintf_l(__s, __n, __loc, __format, std::forward<_Args>(__args)...);
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __libcpp_asprintf_l(
    char** __s, locale_t __loc, const char* __format, _Args&&... __args) {
  return ::asprintf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__scanf__, 3, 4) int __libcpp_sscanf_l(
    const char* __s, locale_t __loc, const char* __format, _Args&&... __args) {
  return ::sscanf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}
_LIBCPP_DIAGNOSTIC_POP
#undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_BSD_LOCALE_DEFAULTS_H
