//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_OTHER_BSD_LIKE_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_OTHER_BSD_LIKE_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__std_mbstate_t.h>
#include <__utility/forward.h>
#if _LIBCPP_HAS_WIDE_CHARACTERS
#  include <wchar.h>
#endif

#include <xlocale.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

inline _LIBCPP_HIDE_FROM_ABI decltype(MB_CUR_MAX) __mb_len_max(__locale_t __loc) { return MB_CUR_MAX_L(__loc); }

#if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI wint_t __btowc(int __c, __locale_t __loc) { return ::btowc_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI int __wctob(wint_t __c, __locale_t __loc) { return ::wctob_l(__c, __loc); }

inline _LIBCPP_HIDE_FROM_ABI size_t
__wcsnrtombs(char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return ::wcsnrtombs_l(__dest, __src, __nwc, __len, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __wcrtomb(char* __s, wchar_t __wc, mbstate_t* __ps, __locale_t __loc) {
  return ::wcrtomb_l(__s, __wc, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return ::mbsnrtowcs_l(__dest, __src, __nms, __len, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbrtowc(wchar_t* __pwc, const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  return ::mbrtowc_l(__pwc, __s, __n, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI int __mbtowc(wchar_t* __pwc, const char* __pmb, size_t __max, __locale_t __loc) {
  return ::mbtowc_l(__pwc, __pmb, __max, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t __mbrlen(const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  return ::mbrlen_l(__s, __n, __ps, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsrtowcs(wchar_t* __dest, const char** __src, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return ::mbsrtowcs_l(__dest, __src, __len, __ps, __loc);
}
#endif

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wgcc-compat")
_LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wformat-nonliteral") // GCC doesn't support [[gnu::format]] on variadic templates
#ifdef _LIBCPP_COMPILER_CLANG_BASED
#  define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(...) _LIBCPP_ATTRIBUTE_FORMAT(__VA_ARGS__)
#else
#  define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(...) /* nothing */
#endif

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 4, 5) int __snprintf(
    char* __s, size_t __n, __locale_t __loc, const char* __format, _Args&&... __args) {
  return ::snprintf_l(__s, __n, __loc, __format, std::forward<_Args>(__args)...);
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __asprintf(
    char** __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  return ::asprintf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__scanf__, 3, 4) int __sscanf(
    const char* __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  return ::sscanf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}
_LIBCPP_DIAGNOSTIC_POP
#undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_OTHER_BSD_LIKE_H
