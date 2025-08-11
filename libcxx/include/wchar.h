// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
    wchar.h synopsis

Macros:

    NULL
    WCHAR_MAX
    WCHAR_MIN
    WEOF

Types:

    mbstate_t
    size_t
    tm
    wint_t

int fwprintf(FILE* restrict stream, const wchar_t* restrict format, ...);
int fwscanf(FILE* restrict stream, const wchar_t* restrict format, ...);
int swprintf(wchar_t* restrict s, size_t n, const wchar_t* restrict format, ...);
int swscanf(const wchar_t* restrict s, const wchar_t* restrict format, ...);
int vfwprintf(FILE* restrict stream, const wchar_t* restrict format, va_list arg);
int vfwscanf(FILE* restrict stream, const wchar_t* restrict format, va_list arg);  // C99
int vswprintf(wchar_t* restrict s, size_t n, const wchar_t* restrict format, va_list arg);
int vswscanf(const wchar_t* restrict s, const wchar_t* restrict format, va_list arg);  // C99
int vwprintf(const wchar_t* restrict format, va_list arg);
int vwscanf(const wchar_t* restrict format, va_list arg);  // C99
int wprintf(const wchar_t* restrict format, ...);
int wscanf(const wchar_t* restrict format, ...);
wint_t fgetwc(FILE* stream);
wchar_t* fgetws(wchar_t* restrict s, int n, FILE* restrict stream);
wint_t fputwc(wchar_t c, FILE* stream);
int fputws(const wchar_t* restrict s, FILE* restrict stream);
int fwide(FILE* stream, int mode);
wint_t getwc(FILE* stream);
wint_t getwchar();
wint_t putwc(wchar_t c, FILE* stream);
wint_t putwchar(wchar_t c);
wint_t ungetwc(wint_t c, FILE* stream);
double wcstod(const wchar_t* restrict nptr, wchar_t** restrict endptr);
float wcstof(const wchar_t* restrict nptr, wchar_t** restrict endptr);         // C99
long double wcstold(const wchar_t* restrict nptr, wchar_t** restrict endptr);  // C99
long wcstol(const wchar_t* restrict nptr, wchar_t** restrict endptr, int base);
long long wcstoll(const wchar_t* restrict nptr, wchar_t** restrict endptr, int base);  // C99
unsigned long wcstoul(const wchar_t* restrict nptr, wchar_t** restrict endptr, int base);
unsigned long long wcstoull(const wchar_t* restrict nptr, wchar_t** restrict endptr, int base);  // C99
wchar_t* wcscpy(wchar_t* restrict s1, const wchar_t* restrict s2);
wchar_t* wcsncpy(wchar_t* restrict s1, const wchar_t* restrict s2, size_t n);
wchar_t* wcscat(wchar_t* restrict s1, const wchar_t* restrict s2);
wchar_t* wcsncat(wchar_t* restrict s1, const wchar_t* restrict s2, size_t n);
int wcscmp(const wchar_t* s1, const wchar_t* s2);
int wcscoll(const wchar_t* s1, const wchar_t* s2);
int wcsncmp(const wchar_t* s1, const wchar_t* s2, size_t n);
size_t wcsxfrm(wchar_t* restrict s1, const wchar_t* restrict s2, size_t n);
const wchar_t* wcschr(const wchar_t* s, wchar_t c);
      wchar_t* wcschr(      wchar_t* s, wchar_t c);
size_t wcscspn(const wchar_t* s1, const wchar_t* s2);
size_t wcslen(const wchar_t* s);
const wchar_t* wcspbrk(const wchar_t* s1, const wchar_t* s2);
      wchar_t* wcspbrk(      wchar_t* s1, const wchar_t* s2);
const wchar_t* wcsrchr(const wchar_t* s, wchar_t c);
      wchar_t* wcsrchr(      wchar_t* s, wchar_t c);
size_t wcsspn(const wchar_t* s1, const wchar_t* s2);
const wchar_t* wcsstr(const wchar_t* s1, const wchar_t* s2);
      wchar_t* wcsstr(      wchar_t* s1, const wchar_t* s2);
wchar_t* wcstok(wchar_t* restrict s1, const wchar_t* restrict s2, wchar_t** restrict ptr);
const wchar_t* wmemchr(const wchar_t* s, wchar_t c, size_t n);
      wchar_t* wmemchr(      wchar_t* s, wchar_t c, size_t n);
int wmemcmp(wchar_t* restrict s1, const wchar_t* restrict s2, size_t n);
wchar_t* wmemcpy(wchar_t* restrict s1, const wchar_t* restrict s2, size_t n);
wchar_t* wmemmove(wchar_t* s1, const wchar_t* s2, size_t n);
wchar_t* wmemset(wchar_t* s, wchar_t c, size_t n);
size_t wcsftime(wchar_t* restrict s, size_t maxsize, const wchar_t* restrict format,
                const tm* restrict timeptr);
wint_t btowc(int c);
int wctob(wint_t c);
int mbsinit(const mbstate_t* ps);
size_t mbrlen(const char* restrict s, size_t n, mbstate_t* restrict ps);
size_t mbrtowc(wchar_t* restrict pwc, const char* restrict s, size_t n, mbstate_t* restrict ps);
size_t wcrtomb(char* restrict s, wchar_t wc, mbstate_t* restrict ps);
size_t mbsrtowcs(wchar_t* restrict dst, const char** restrict src, size_t len,
                 mbstate_t* restrict ps);
size_t wcsrtombs(char* restrict dst, const wchar_t** restrict src, size_t len,
                 mbstate_t* restrict ps);

*/

// #include_next has different semantics for module builds.
// include_next<wchar.h> only works if the current filename is wchar.h,
// otherwise it just does a regular #include.
// To solve this, if _LIBCPP_INCLUDE_NEXT_WCHAR is defined, fake an include_next.
#ifdef _LIBCPP_INCLUDE_NEXT_WCHAR
#  if __has_include_next(<wchar.h>)
#    include_next <wchar.h>
#  elif defined(_LIBCPP_INCLUDE_NEXT_WCHAR)
#    define _LIBCPP_WCHAR_NOT_FOUND
#  endif
#elif defined(__cplusplus) && __cplusplus < 201103L && defined(_LIBCPP_USE_FROZEN_CXX03_HEADERS)
#  include <__cxx03/wchar.h>
#else
#  include <__config>

#  if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#    pragma GCC system_header
#  endif

// We define this here to support older versions of glibc <wchar.h> that do
// not define this for clang.
#  if defined(__cplusplus) && !defined(__CORRECT_ISO_CPP_WCHAR_H_PROTO)
#    define __CORRECT_ISO_CPP_WCHAR_H_PROTO
#  endif

// The inclusion of the system's <wchar.h> is intentionally done once outside of any include
// guards because some code expects to be able to include the underlying system header multiple
// times to get different definitions based on the macros that are set before inclusion.
#  if __has_include_next(<wchar.h>)
#    include_next <wchar.h>
#  endif

// Place the header guard here to make it visible to cwchar.
#  if !defined(_LIBCPP_WCHAR_H)
#    define _LIBCPP_WCHAR_H
// This section is not safe to include multiple times, so it goes in a seperate
// file which is marked as non textual in the modulemap.
#    include <__wchar.h>
#  endif

#endif // defined(__cplusplus) && __cplusplus < 201103L && defined(_LIBCPP_USE_FROZEN_CXX03_HEADERS)
