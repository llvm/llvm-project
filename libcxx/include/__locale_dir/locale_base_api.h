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
// Most of these functions are only required when building the library. Functions that are also
// required when merely using the headers are marked as such below.
//
// TODO: __localeconv shouldn't take a reference, but the Windows implementation doesn't allow copying __locale_t
// TODO: Eliminate the need for any of these functions from the headers.
//
// Locale management
// -----------------
// namespace __locale {
//  using __locale_t = implementation-defined;  // required by the headers
//  using __lconv_t  = implementation-defined;
//  __locale_t  __newlocale(int, const char*, __locale_t);
//  void        __freelocale(__locale_t);
//  char*       __setlocale(int, const char*);
//  __lconv_t*  __localeconv(__locale_t&);
// }
//
// // required by the headers
// #define _LIBCPP_COLLATE_MASK   /* implementation-defined */
// #define _LIBCPP_CTYPE_MASK     /* implementation-defined */
// #define _LIBCPP_MONETARY_MASK  /* implementation-defined */
// #define _LIBCPP_NUMERIC_MASK   /* implementation-defined */
// #define _LIBCPP_TIME_MASK      /* implementation-defined */
// #define _LIBCPP_MESSAGES_MASK  /* implementation-defined */
// #define _LIBCPP_ALL_MASK       /* implementation-defined */
// #define _LIBCPP_LC_ALL         /* implementation-defined */
//
// Strtonum functions
// ------------------
// namespace __locale {
//  // required by the headers
//  float               __strtof(const char*, char**, __locale_t);
//  double              __strtod(const char*, char**, __locale_t);
//  long double         __strtold(const char*, char**, __locale_t);
// }
//
// Character manipulation functions
// --------------------------------
// namespace __locale {
//  int     __toupper(int, __locale_t);
//  int     __tolower(int, __locale_t);
//  int     __strcoll(const char*, const char*, __locale_t);
//  size_t  __strxfrm(char*, const char*, size_t, __locale_t);
//
//  int     __iswctype(wint_t, wctype_t, __locale_t);
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
//
//  int     __snprintf(char*, size_t, __locale_t, const char*, ...); // required by the headers
//  int     __asprintf(char**, __locale_t, const char*, ...);        // required by the headers
// }

#if _LIBCPP_HAS_LOCALIZATION

#  if defined(__APPLE__)
#    include <__locale_dir/support/apple.h>
#  elif defined(__FreeBSD__)
#    include <__locale_dir/support/freebsd.h>
#  elif defined(__NetBSD__)
#    include <__locale_dir/support/netbsd.h>
#  elif defined(__OpenBSD__)
#    include <__locale_dir/support/openbsd.h>
#  elif defined(_LIBCPP_MSVCRT_LIKE)
#    include <__locale_dir/support/windows.h>
#  elif defined(__Fuchsia__)
#    include <__locale_dir/support/fuchsia.h>
#  elif _LIBCPP_LIBC_LLVM_LIBC
#    include <__locale_dir/support/llvm_libc.h>
#  elif defined(__linux__)
#    include <__locale_dir/support/linux.h>
#  elif _LIBCPP_LIBC_NEWLIB
#    include <__locale_dir/support/newlib.h>
#  elif defined(_AIX)
#    include <__locale_dir/support/aix.h>
#  elif defined(__MVS__)
#    include <__locale_dir/support/mvs.h>
#  else
#    warning "No known way to provide the locale base API"
#  endif // Compatibility definition of locale base APIs

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___LOCALE_DIR_LOCALE_BASE_API_H
