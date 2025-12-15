//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_AIX_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_AIX_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__std_mbstate_t.h>
#include <__utility/forward.h>
#include <clocale> // std::lconv
#include <cstdio>
#include <cstdlib>
#include <ctype.h>
#include <string.h>
#include <time.h>
#if _LIBCPP_HAS_WIDE_CHARACTERS
#  include <cwchar>
#  include <wctype.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

struct __locale_guard {
  _LIBCPP_HIDE_FROM_ABI __locale_guard(locale_t& __loc) : __old_loc_(::uselocale(__loc)) {}

  _LIBCPP_HIDE_FROM_ABI ~__locale_guard() {
    if (__old_loc_)
      ::uselocale(__old_loc_);
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

// Get MB_CUE_MAX before __locale_t is defined because it uses a defferent definition in <sys/localedef.h>.
static inline _LIBCPP_HIDE_FROM_ABI decltype(MB_CUR_MAX) __mb_cur_max() { return MB_CUR_MAX; }

using __locale_t _LIBCPP_NODEBUG = ::locale_t;

#if defined(_LIBCPP_BUILDING_LIBRARY)
using __lconv_t _LIBCPP_NODEBUG = std::lconv;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __locale, __locale_t __base) {
  return ::newlocale(__category_mask, __locale, __base);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { ::freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI char* __setlocale(int __category, char const* __locale) {
  return ::setlocale(__category, __locale);
}

inline _LIBCPP_HIDE_FROM_ABI __lconv_t* __localeconv(__locale_t& __loc) {
  __locale_guard __current(__loc);
  return std::localeconv();
}
#endif // _LIBCPP_BUILDING_LIBRARY

// The following structure is a quick-and-dirty workaround for routines that AIX
// does not provide in the "_l" (locale-aware) variants.
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

//
// Strtonum functions
//
inline _LIBCPP_HIDE_FROM_ABI float __strtof(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtof(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI double __strtod(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtod(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long double __strtold(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtold(__nptr, __endptr);
}

//
// Character manipulation functions
//
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
inline _LIBCPP_HIDE_FROM_ABI decltype(__mb_cur_max()) __mb_len_max(__locale_t __loc) {
  __locale_guard __current(__loc);
  return __mb_cur_max();
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
  return ::wcsnrtombs(__dest, __src, __nwc, __len, __ps); // non-standard
}

inline _LIBCPP_HIDE_FROM_ABI size_t __wcrtomb(char* __s, wchar_t __wc, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wcrtomb(__s, __wc, __ps);
}

inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return ::mbsnrtowcs(__dest, __src, __nms, __len, __ps); // non-standard
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
#  endif // _LIBCPP_HAS_WIDE_CHARACTERS
#endif   // _LIBCPP_BUILDING_LIBRARY

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
  __locale_guard __current(__loc);
  return std::snprintf(__s, __n, __format, std::forward<_Args>(__args)...);
}

template <class... _Args>
inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __asprintf(
    char** __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  __locale_guard __current(__loc);

  // Probe exact size.
  int n = std::snprintf(nullptr, 0, __format, std::forward<_Args>(__args)...);
  if (n < 0) {
    *__s = nullptr;
    return -1;
  }

  // Allocate and render once
  size_t buf_size = static_cast<size_t>(n) + 1;
  char* buf       = static_cast<char*>(std::malloc(buf_size));
  if (!buf) {
    *__s = nullptr;
    return -1;
  }

  int written_size = std::snprintf(buf, buf_size, __format, std::forward<_Args>(__args)...);
  if (written_size < 0) {
    std::free(buf);
    *__s = nullptr;
    return -1;
  }

  *__s = buf;
  return written_size;
}
_LIBCPP_DIAGNOSTIC_POP
#undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_AIX_H
