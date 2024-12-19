//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_H
#define _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

// The platform-specific headers have to provide the following interface.
//
// These functions are equivalent to their C counterparts, except that __locale::__locale_t
// is used instead of the current global locale.
//
// Variadic functions may be implemented as templates with a parameter pack instead
// of C-style variadic functions.
//
// TODO: __localeconv shouldn't take a reference, but the Windows implementation doesn't allow copying __locale_t
//
// Locale management
// -----------------
// namespace __locale {
//  using __locale_t = implementation-defined;
//  __locale_t  __newlocale(int, const char*, __locale_t);
//  void        __freelocale(__locale_t);
//  lconv*      __localeconv(__locale_t&);
// }
//
// Strtonum functions
// ------------------
// namespace __locale {
//  float               __strtof(const char*, char**, __locale_t);
//  double              __strtod(const char*, char**, __locale_t);
//  long double         __strtold(const char*, char**, __locale_t);
//  long long           __strtoll(const char*, char**, __locale_t);
//  unsigned long long  __strtoull(const char*, char**, __locale_t);
// }
//
// Character manipulation functions
// --------------------------------
// namespace __locale {
//  int     __islower(int, __locale_t);
//  int     __isupper(int, __locale_t);
//  int     __isdigit(int, __locale_t);
//  int     __isxdigit(int, __locale_t);
//  int     __toupper(int, __locale_t);
//  int     __tolower(int, __locale_t);
//  int     __strcoll(const char*, const char*, __locale_t);
//  size_t  __strxfrm(char*, const char*, size_t, __locale_t);
//
//  int     __iswspace(wint_t, __locale_t);
//  int     __iswprint(wint_t, __locale_t);
//  int     __iswcntrl(wint_t, __locale_t);
//  int     __iswupper(wint_t, __locale_t);
//  int     __iswlower(wint_t, __locale_t);
//  int     __iswalpha(wint_t, __locale_t);
//  int     __iswblank(wint_t, __locale_t);
//  int     __iswdigit(wint_t, __locale_t);
//  int     __iswpunct(wint_t, __locale_t);
//  int     __iswxdigit(wint_t, __locale_t);
//  wint_t  __towupper(wint_t, __locale_t);
//  wint_t  __towlower(wint_t, __locale_t);
//  int     __wcscoll(const wchar_t*, const wchar_t*, __locale_t);
//  size_t  __wcsxfrm(wchar_t*, const wchar_t*, size_t, __locale_t);
//
//  size_t  __strftime(char*, size_t, const char*, const tm*, __locale_t);
// }
//
// Other functions
// ---------------
// namespace __locale {
//  implementation-defined __mb_len_max(__locale_t);
//  wint_t  __btowc(int, __locale_t);
//  int     __wctob(wint_t, __locale_t);
//  size_t  __wcsnrtombs(char*, const wchar_t**, size_t, size_t, mbstate_t*, __locale_t);
//  size_t  __wcrtomb(char*, wchar_t, mbstate_t*, __locale_t);
//  size_t  __mbsnrtowcs(wchar_t*, const char**, size_t, size_t, mbstate_t*, __locale_t);
//  size_t  __mbrtowc(wchar_t*, const char*, size_t, mbstate_t*, __locale_t);
//  int     __mbtowc(wchar_t*, const char*, size_t, __locale_t);
//  size_t  __mbrlen(const char*, size_t, mbstate_t*, __locale_t);
//  size_t  __mbsrtowcs(wchar_t*, const char**, size_t, mbstate_t*, __locale_t);
//  int     __snprintf(char*, size_t, __locale_t, const char*, ...);
//  int     __asprintf(char**, __locale_t, const char*, ...);
//  int     __sscanf(const char*, __locale_t, const char*, ...);
// }

#if defined(__APPLE__)
#  include <__locale_dir/support/apple.h>
#elif defined(__FreeBSD__)
#  include <__locale_dir/support/freebsd.h>
#elif defined(_LIBCPP_MSVCRT_LIKE)
#  include <__locale_dir/support/windows.h>
#else

// TODO: This is a temporary definition to bridge between the old way we defined the locale base API
//       (by providing global non-reserved names) and the new API. As we move individual platforms
//       towards the new way of defining the locale base API, this should disappear since each platform
//       will define those directly.
#  if defined(_AIX) || defined(__MVS__)
#    include <__locale_dir/locale_base_api/ibm.h>
#  elif defined(__ANDROID__)
#    include <__locale_dir/locale_base_api/android.h>
#  elif defined(__OpenBSD__)
#    include <__locale_dir/locale_base_api/openbsd.h>
#  elif defined(__Fuchsia__)
#    include <__locale_dir/locale_base_api/fuchsia.h>
#  elif defined(__wasi__) || _LIBCPP_HAS_MUSL_LIBC
#    include <__locale_dir/locale_base_api/musl.h>
#  endif

#  ifdef _LIBCPP_LOCALE__L_EXTENSIONS
#    include <__locale_dir/locale_base_api/bsd_locale_defaults.h>
#  else
#    include <__locale_dir/locale_base_api/bsd_locale_fallbacks.h>
#  endif

#  include <__cstddef/size_t.h>
#  include <__utility/forward.h>
#  include <ctype.h>
#  include <string.h>
#  include <time.h>
#  if _LIBCPP_HAS_WIDE_CHARACTERS
#    include <wctype.h>
#  endif
_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {
//
// Locale management
//
using __locale_t = locale_t;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __name, __locale_t __loc) {
  return newlocale(__category_mask, __name, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI lconv* __localeconv(__locale_t& __loc) { return __libcpp_localeconv_l(__loc); }

//
// Strtonum functions
//
inline _LIBCPP_HIDE_FROM_ABI float __strtof(const char* __nptr, char** __endptr, __locale_t __loc) {
  return strtof_l(__nptr, __endptr, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI double __strtod(const char* __nptr, char** __endptr, __locale_t __loc) {
  return strtod_l(__nptr, __endptr, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI long double __strtold(const char* __nptr, char** __endptr, __locale_t __loc) {
  return strtold_l(__nptr, __endptr, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI long long __strtoll(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  return strtoll_l(__nptr, __endptr, __base, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI unsigned long long
__strtoull(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  return strtoull_l(__nptr, __endptr, __base, __loc);
}

//
// Character manipulation functions
//
inline _LIBCPP_HIDE_FROM_ABI int __islower(int __ch, __locale_t __loc) { return islower_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __isupper(int __ch, __locale_t __loc) { return isupper_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __isdigit(int __ch, __locale_t __loc) { return isdigit_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __isxdigit(int __ch, __locale_t __loc) { return isxdigit_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __strcoll(const char* __s1, const char* __s2, __locale_t __loc) {
  return strcoll_l(__s1, __s2, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __strxfrm(char* __dest, const char* __src, size_t __n, __locale_t __loc) {
  return strxfrm_l(__dest, __src, __n, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI int __toupper(int __ch, __locale_t __loc) { return toupper_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __tolower(int __ch, __locale_t __loc) { return tolower_l(__ch, __loc); }

#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI int __wcscoll(const wchar_t* __s1, const wchar_t* __s2, __locale_t __loc) {
  return wcscoll_l(__s1, __s2, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __wcsxfrm(wchar_t* __dest, const wchar_t* __src, size_t __n, __locale_t __loc) {
  return wcsxfrm_l(__dest, __src, __n, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI int __iswspace(wint_t __ch, __locale_t __loc) { return iswspace_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswprint(wint_t __ch, __locale_t __loc) { return iswprint_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswcntrl(wint_t __ch, __locale_t __loc) { return iswcntrl_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswupper(wint_t __ch, __locale_t __loc) { return iswupper_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswlower(wint_t __ch, __locale_t __loc) { return iswlower_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswalpha(wint_t __ch, __locale_t __loc) { return iswalpha_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswblank(wint_t __ch, __locale_t __loc) { return iswblank_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswdigit(wint_t __ch, __locale_t __loc) { return iswdigit_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswpunct(wint_t __ch, __locale_t __loc) { return iswpunct_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __iswxdigit(wint_t __ch, __locale_t __loc) { return iswxdigit_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI wint_t __towupper(wint_t __ch, __locale_t __loc) { return towupper_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI wint_t __towlower(wint_t __ch, __locale_t __loc) { return towlower_l(__ch, __loc); }
#  endif

inline _LIBCPP_HIDE_FROM_ABI size_t
__strftime(char* __s, size_t __max, const char* __format, const tm* __tm, __locale_t __loc) {
  return strftime_l(__s, __max, __format, __tm, __loc);
}

//
// Other functions
//
inline _LIBCPP_HIDE_FROM_ABI decltype(__libcpp_mb_cur_max_l(__locale_t())) __mb_len_max(__locale_t __loc) {
  return __libcpp_mb_cur_max_l(__loc);
}
#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI wint_t __btowc(int __ch, __locale_t __loc) { return __libcpp_btowc_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI int __wctob(wint_t __ch, __locale_t __loc) { return __libcpp_wctob_l(__ch, __loc); }
inline _LIBCPP_HIDE_FROM_ABI size_t
__wcsnrtombs(char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_wcsnrtombs_l(__dest, __src, __nwc, __len, __ps, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __wcrtomb(char* __s, wchar_t __ch, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_wcrtomb_l(__s, __ch, __ps, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_mbsnrtowcs_l(__dest, __src, __nms, __len, __ps, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t
__mbrtowc(wchar_t* __pwc, const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_mbrtowc_l(__pwc, __s, __n, __ps, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI int __mbtowc(wchar_t* __pwc, const char* __pmb, size_t __max, __locale_t __loc) {
  return __libcpp_mbtowc_l(__pwc, __pmb, __max, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __mbrlen(const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_mbrlen_l(__s, __n, __ps, __loc);
}
inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsrtowcs(wchar_t* __dest, const char** __src, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  return __libcpp_mbsrtowcs_l(__dest, __src, __len, __ps, __loc);
}
#  endif

_LIBCPP_DIAGNOSTIC_PUSH
_LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wgcc-compat")
_LIBCPP_GCC_DIAGNOSTIC_IGNORED("-Wformat-nonliteral") // GCC doesn't support [[gnu::format]] on variadic templates
#  ifdef _LIBCPP_COMPILER_CLANG_BASED
#    define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(...) _LIBCPP_ATTRIBUTE_FORMAT(__VA_ARGS__)
#  else
#    define _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(...) /* nothing */
#  endif

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 4, 5) int __snprintf(
    char* __s, size_t __n, __locale_t __loc, const char* __format, _Args&&... __args) {
  return std::__libcpp_snprintf_l(__s, __n, __loc, __format, std::forward<_Args>(__args)...);
}
template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __asprintf(
    char** __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  return std::__libcpp_asprintf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}
template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__scanf__, 3, 4) int __sscanf(
    const char* __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  return std::__libcpp_sscanf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}
_LIBCPP_DIAGNOSTIC_POP
#  undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // Compatibility definition of locale base APIs

#endif // _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_H
