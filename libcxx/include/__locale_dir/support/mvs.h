// -*- C++ -*-
//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_MVS_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_MVS_H

#include <__config>
#include <__cstddef/size_t.h>
#include <__utility/forward.h>
#include <cstdlib>
#include <ctype.h>
#include <locale.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <wctype.h>

#if _LIBCPP_HAS_WIDE_CHARACTERS
#  include <wchar.h>
#  include <wctype.h>
#endif

#include <string>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

typedef struct locale_struct {
  int category_mask;
  std::string lc_collate;
  std::string lc_ctype;
  std::string lc_monetary;
  std::string lc_numeric;
  std::string lc_time;
  std::string lc_messages;
}* locale_t;

// z/OS does not have newlocale, freelocale and uselocale.
// The functions below are workarounds in single thread mode.
locale_t newlocale(int category_mask, const char* locale, locale_t base);
void freelocale(locale_t locobj);
locale_t uselocale(locale_t newloc);

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

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

struct __locale_guard {
  __locale_guard(locale_t& __loc) : __old_loc_(::uselocale(__loc)) {}

  ~__locale_guard() {
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

#define _LC_MAX LC_MESSAGES /* highest real category */
#define _NCAT (_LC_MAX + 1) /* maximum + 1 */
#define _CATMASK(n) (1 << (n))

#define _LIBCPP_COLLATE_MASK _CATMASK(LC_COLLATE)
#define _LIBCPP_CTYPE_MASK _CATMASK(LC_CTYPE)
#define _LIBCPP_MONETARY_MASK _CATMASK(LC_MONETARY)
#define _LIBCPP_NUMERIC_MASK _CATMASK(LC_NUMERIC)
#define _LIBCPP_TIME_MASK _CATMASK(LC_TIME)
#define _LIBCPP_MESSAGES_MASK _CATMASK(LC_MESSAGES)
#define _LIBCPP_ALL_MASK (_CATMASK(_NCAT) - 1)
#define _LIBCPP_LC_ALL LC_ALL

using __locale_t _LIBCPP_NODEBUG = locale_t;

#if defined(_LIBCPP_BUILDING_LIBRARY)
using __lconv_t _LIBCPP_NODEBUG = lconv;

inline __locale_t __newlocale(int __category_mask, const char* __name, __locale_t __loc) {
  return newlocale(__category_mask, __name, __loc);
}

inline char* __setlocale(int __category, char const* __locale) { return ::setlocale(__category, __locale); }

inline void __freelocale(__locale_t __loc) { freelocale(__loc); }

inline __lconv_t* __localeconv(__locale_t& __loc) {
  __locale_guard __current(__loc);
  return localeconv();
}
#endif // _LIBCPP_BUILDING_LIBRARY

//
// Strtonum functions
//
inline float __strtof(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtof(__nptr, __endptr);
}

inline double __strtod(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtod(__nptr, __endptr);
}

inline long double __strtold(const char* __nptr, char** __endptr, __locale_t __loc) {
  __setAndRestore __newloc(__loc);
  return ::strtold(__nptr, __endptr);
}

//
// Character manipulation functions
//
#if defined(_LIBCPP_BUILDING_LIBRARY)
inline int __strcoll(const char* __s1, const char* __s2, __locale_t __loc) { return ::strcoll(__s1, __s2); }

inline size_t __strxfrm(char* __dest, const char* __src, size_t __n, __locale_t __loc) {
  return ::strxfrm(__dest, __src, __n);
}

inline int __toupper(int __ch, __locale_t __loc) { return ::toupper(__ch); }
inline int __tolower(int __ch, __locale_t __loc) { return ::tolower(__ch); }

#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline int __wcscoll(const wchar_t* __s1, const wchar_t* __s2, __locale_t __loc) { return ::wcscoll(__s1, __s2); }

inline size_t __wcsxfrm(wchar_t* __dest, const wchar_t* __src, size_t __n, __locale_t __loc) {
  return ::wcsxfrm(__dest, __src, __n);
}

inline int __iswctype(wint_t __ch, wctype_t __type, __locale_t __loc) { return ::iswctype(__ch, __type); }
inline int __iswspace(wint_t __ch, __locale_t __loc) { return ::iswspace(__ch); }
inline int __iswprint(wint_t __ch, __locale_t __loc) { return ::iswprint(__ch); }
inline int __iswcntrl(wint_t __ch, __locale_t __loc) { return ::iswcntrl(__ch); }
inline int __iswupper(wint_t __ch, __locale_t __loc) { return ::iswupper(__ch); }
inline int __iswlower(wint_t __ch, __locale_t __loc) { return ::iswlower(__ch); }
inline int __iswalpha(wint_t __ch, __locale_t __loc) { return ::iswalpha(__ch); }
inline int __iswblank(wint_t __ch, __locale_t __loc) { return ::iswblank(__ch); }
inline int __iswdigit(wint_t __ch, __locale_t __loc) { return ::iswdigit(__ch); }
inline int __iswpunct(wint_t __ch, __locale_t __loc) { return ::iswpunct(__ch); }
inline int __iswxdigit(wint_t __ch, __locale_t __loc) { return ::iswxdigit(__ch); }
inline wint_t __towupper(wint_t __ch, __locale_t __loc) { return ::towupper(__ch); }
inline wint_t __towlower(wint_t __ch, __locale_t __loc) { return ::towlower(__ch); }
#  endif

inline _LIBCPP_ATTRIBUTE_FORMAT(__strftime__, 3, 0) size_t
    __strftime(char* __s, size_t __max, const char* __format, const tm* __tm, __locale_t __loc) {
  return ::strftime(__s, __max, __format, __tm);
}

//
// Other functions
//
inline decltype(MB_CUR_MAX) __mb_len_max(__locale_t __loc) {
  __locale_guard __current(__loc);
  return MB_CUR_MAX;
}
#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline wint_t __btowc(int __ch, __locale_t __loc) {
  __locale_guard __current(__loc);
  return btowc(__ch);
}

inline int __wctob(wint_t __ch, __locale_t __loc) {
  __locale_guard __current(__loc);
  return wctob(__ch);
}

inline size_t
__wcsnrtombs(char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return wcsnrtombs(__dest, __src, __nwc, __len, __ps);
}

inline size_t __wcrtomb(char* __s, wchar_t __ch, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return wcrtomb(__s, __ch, __ps);
}

inline size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbsnrtowcs(__dest, __src, __nms, __len, __ps);
}

inline size_t __mbrtowc(wchar_t* __pwc, const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbrtowc(__pwc, __s, __n, __ps);
}

inline int __mbtowc(wchar_t* __pwc, const char* __pmb, size_t __max, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbtowc(__pwc, __pmb, __max);
}

inline size_t __mbrlen(const char* __s, size_t __n, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbrlen(__s, __n, __ps);
}

inline size_t __mbsrtowcs(wchar_t* __dest, const char** __src, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return mbsrtowcs(__dest, __src, __len, __ps);
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

// FIXME: This should be inlined
inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 4, 5) int __libcpp_snprintf_l(
    char* __s, size_t __n, locale_t __l, const char* __format, ...) {
  va_list __va;
  va_start(__va, __format);
  __locale_guard __current(__l);
  int __res = vsnprintf(__s, __n, __format, __va);
  va_end(__va);
  return __res;
}

template <class... _Args>
_LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 4, 5)
int __snprintf(char* __s, size_t __n, __locale_t __loc, const char* __format, _Args&&... __args) {
  return __locale::__libcpp_snprintf_l(__s, __n, __loc, __format, std::forward<_Args>(__args)...);
}

// FIXME: This should be inlined
inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 2, 0) int vasprintf(char** strp, const char* fmt, va_list ap) {
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

// FIXME: This should be inlined
inline _LIBCPP_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __libcpp_asprintf_l(
    char** __s, locale_t __l, const char* __format, ...) {
  va_list __va;
  va_start(__va, __format);
  __locale_guard __current(__l);
  int __res = vasprintf(__s, __format, __va);
  va_end(__va);
  return __res;
}

template <class... _Args>
_LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4)
int __asprintf(char** __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  return __locale::__libcpp_asprintf_l(__s, __loc, __format, std::forward<_Args>(__args)...);
}
_LIBCPP_DIAGNOSTIC_POP
#undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT

} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_MVS_H
