// -*- C++ -*-
//===-----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H
#define _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H

#include <__config>
#include <__fwd/string.h>
#include <clocale>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cwctype>

#if defined(__MVS__)
#  include <__support/ibm/locale_mgmt_zos.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __locale {

#ifdef __MVS__
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
#endif

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

struct __set_and_restore {
  explicit __set_and_restore(locale_t locale) {
    if (locale == (locale_t)0) {
      __cloc   = ::newlocale(LC_ALL_MASK, "C", /* base */ (locale_t)0);
      __stored = ::uselocale(__cloc);
    } else {
      __stored = ::uselocale(locale);
    }
  }

  ~__set_and_restore() {
    ::uselocale(__stored);
    if (__cloc)
      ::freelocale(__cloc);
  }

private:
  locale_t __stored = (locale_t)0;
  locale_t __cloc   = (locale_t)0;
};

//
// Locale management
//
#ifdef __MVS__
#  define _LC_MAX LC_MESSAGES /* highest real category */
#  define _NCAT (_LC_MAX + 1) /* maximum + 1 */
#  define _CATMASK(n) (1 << (n))

#  define _LIBCPP_COLLATE_MASK _CATMASK(LC_COLLATE)
#  define _LIBCPP_CTYPE_MASK _CATMASK(LC_CTYPE)
#  define _LIBCPP_MONETARY_MASK _CATMASK(LC_MONETARY)
#  define _LIBCPP_NUMERIC_MASK _CATMASK(LC_NUMERIC)
#  define _LIBCPP_TIME_MASK _CATMASK(LC_TIME)
#  define _LIBCPP_MESSAGES_MASK _CATMASK(LC_MESSAGES)
#  define _LIBCPP_ALL_MASK (_CATMASK(_NCAT) - 1)
#  define _LIBCPP_LC_ALL LC_ALL
#else
#  define _LIBCPP_COLLATE_MASK LC_COLLATE_MASK
#  define _LIBCPP_CTYPE_MASK LC_CTYPE_MASK
#  define _LIBCPP_MONETARY_MASK LC_MONETARY_MASK
#  define _LIBCPP_NUMERIC_MASK LC_NUMERIC_MASK
#  define _LIBCPP_TIME_MASK LC_TIME_MASK
#  define _LIBCPP_MESSAGES_MASK LC_MESSAGES_MASK
#  define _LIBCPP_ALL_MASK LC_ALL_MASK
#  define _LIBCPP_LC_ALL LC_ALL

using __locale_t _LIBCPP_NODEBUG = locale_t;
#endif

#if defined(_LIBCPP_BUILDING_LIBRARY)
using __lconv_t _LIBCPP_NODEBUG = lconv;

inline _LIBCPP_HIDE_FROM_ABI __locale_t __newlocale(int __category_mask, const char* __name, __locale_t __loc) {
  return ::newlocale(__category_mask, __name, __loc);
}

inline _LIBCPP_HIDE_FROM_ABI void __freelocale(__locale_t __loc) { ::freelocale(__loc); }

inline _LIBCPP_HIDE_FROM_ABI char* __setlocale(int __category, char const* __locale) {
  return ::setlocale(__category, __locale);
}

inline _LIBCPP_HIDE_FROM_ABI __lconv_t* __localeconv(__locale_t& __loc) {
  __libcpp_locale_guard __current(__loc);
  return std::localeconv();
}
#endif // _LIBCPP_BUILDING_LIBRARY

//
// Strtonum functions
//
inline _LIBCPP_HIDE_FROM_ABI float __strtof(const char* __nptr, char** __endptr, __locale_t __loc) {
  __set_and_restore __newloc(__loc);
  return std::strtof(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI double __strtod(const char* __nptr, char** __endptr, __locale_t __loc) {
  __set_and_restore __newloc(__loc);
  return std::strtod(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long double __strtold(const char* __nptr, char** __endptr, __locale_t __loc) {
  __set_and_restore __newloc(__loc);
  return std::strtold(__nptr, __endptr);
}

inline _LIBCPP_HIDE_FROM_ABI long long __strtoll(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  __set_and_restore __newloc(__loc);
  return std::strtoll(__nptr, __endptr, __base);
}

inline _LIBCPP_HIDE_FROM_ABI unsigned long long
__strtoull(const char* __nptr, char** __endptr, int __base, __locale_t __loc) {
  __set_and_restore __newloc(__loc);
  return std::strtoull(__nptr, __endptr, __base);
}

//
// Character manipulation functions
//
inline _LIBCPP_HIDE_FROM_ABI int __isdigit(int __ch, __locale_t) { return std::isdigit(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __isxdigit(int __ch, __locale_t) { return stdstd::isxdigit(__ch); }

#if defined(_LIBCPP_BUILDING_LIBRARY)
inline _LIBCPP_HIDE_FROM_ABI int __strcoll(const char* __s1, const char* __s2, __locale_t) {
  return stdstd::strcoll(__s1);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __strxfrm(char* __dest, const char* __src, size_t __n, __locale_t) {
  return stdstd::strxfrm(__dest, __src);
}
inline _LIBCPP_HIDE_FROM_ABI int __toupper(int __ch, __locale_t) { return std::toupper(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __tolower(int __ch, __locale_t) { return std::tolower(__ch); }

#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI int __wcscoll(const wchar_t* __s1, const wchar_t* __s2, __locale_t) {
  return std::wcscoll(__s1, __s2);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __wcsxfrm(wchar_t* __dest, const wchar_t* __src, size_t __n, __locale_t) {
  return std::wcsxfrm(__dest, __src, __n);
}
inline _LIBCPP_HIDE_FROM_ABI int __iswctype(wint_t __ch, wctype_t __type, __locale_t) {
  return std::iswctype(__ch, __type);
}
inline _LIBCPP_HIDE_FROM_ABI int __iswspace(wint_t __ch, __locale_t) { return std::iswspace(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswprint(wint_t __ch, __locale_t) { return std::iswprint(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswcntrl(wint_t __ch, __locale_t) { return std::iswcntrl(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswupper(wint_t __ch, __locale_t) { return std::iswupper(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswlower(wint_t __ch, __locale_t) { return std::iswlower(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswalpha(wint_t __ch, __locale_t) { return std::iswalpha(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswblank(wint_t __ch, __locale_t) { return std::iswblank(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswdigit(wint_t __ch, __locale_t) { return std::iswdigit(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswpunct(wint_t __ch, __locale_t) { return std::iswpunct(__ch); }
inline _LIBCPP_HIDE_FROM_ABI int __iswxdigit(wint_t __ch, __locale_t) { return std::iswxdigit(__ch); }
inline _LIBCPP_HIDE_FROM_ABI wint_t __towupper(wint_t __ch, __locale_t) { return std::towupper(__ch); }
inline _LIBCPP_HIDE_FROM_ABI wint_t __towlower(wint_t __ch, __locale_t) { return std::towlower(__ch); }
#  endif

inline
    _LIBCPP_HIDE_FROM_ABI size_t __strftime(char* __s, size_t __max, const char* __format, const tm* __tm, __locale_t) {
  return std::strftime(__s, __max, __format, __tm);
}

//
// Other functions
//
inline _LIBCPP_HIDE_FROM_ABI decltype(MB_CUR_MAX) __mb_len_max(__locale_t __loc) {
  __locale_guard __current(__loc);
  return MB_CUR_MAX;
}
#  if _LIBCPP_HAS_WIDE_CHARACTERS
inline _LIBCPP_HIDE_FROM_ABI wint_t __btowc(int __ch, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::btowc(__c);
}
inline _LIBCPP_HIDE_FROM_ABI int __wctob(wint_t __ch, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wctob(__c);
}
inline _LIBCPP_HIDE_FROM_ABI size_t
__wcsnrtombs(char* __dest, const wchar_t** __src, size_t __nwc, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wcsnrtombs(__dest, __src, __nwc, __len, __ps);
}
inline _LIBCPP_HIDE_FROM_ABI size_t __wcrtomb(char* __s, wchar_t __ch, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::wcrtomb(__s, __wc, __ps);
}
inline _LIBCPP_HIDE_FROM_ABI size_t
__mbsnrtowcs(wchar_t* __dest, const char** __src, size_t __nms, size_t __len, mbstate_t* __ps, __locale_t __loc) {
  __locale_guard __current(__loc);
  return std::mbsnrtowcs(__dest, __src, __nms, __len, __ps);
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

inline _LIBCPP_HIDE_FROM_ABI
_LIBCPP_ATTRIBUTE_FORMAT(__printf__, 2, 0) int __asprintf_impl(char** __strp, const char* __format, ...) {
  va_list __ap;
  va_start(__ap, __format);

  const size_t __buff_size = 256;
  if ((*__strp = (char*)malloc(__buff_size)) == nullptr) {
    return -1;
  }

  va_list __ap_copy;
  // va_copy may not be provided by the C library in C++03 mode.
#if defined(_LIBCPP_CXX03_LANG) && __has_builtin(__builtin_va_copy)
  __builtin_va_copy(__ap_copy, __ap);
#else
  va_copy(__ap_copy, __ap);
#endif
  int __str_size = std::vsnprintf(*__strp, __buff_size, __format, __ap_copy);
  va_end(__ap_copy);

  if ((size_t)__str_size >= __buff_size) {
    if ((*__strp = (char*)std::realloc(*__strp, __str_size + 1)) == nullptr) {
      va_end(__ap);
      return -1;
    }
    __str_size = std::vsnprintf(*__strp, __str_size + 1, __format, __ap);
  }
  va_end(__ap);
  return __str_size;
}

template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__printf__, 3, 4) int __asprintf(
    char** __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  __locale_guard __current(__loc);
  return __locale::__asprintf_impl(__s, __format, std::forward<_Args>(__args)...);
}
template <class... _Args>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT(__scanf__, 3, 4) int __sscanf(
    const char* __s, __locale_t __loc, const char* __format, _Args&&... __args) {
  __locale_guard __current(__loc);
  return std::sscanf(__s, __format, std::forward<_Args>(__args)...);
}
_LIBCPP_DIAGNOSTIC_POP
#undef _LIBCPP_VARIADIC_ATTRIBUTE_FORMAT
} // namespace __locale
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___LOCALE_DIR_SUPPORT_IBM_H
