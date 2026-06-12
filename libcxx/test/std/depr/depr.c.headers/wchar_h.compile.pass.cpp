//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-wide-characters

// <wchar.h>

#include <wchar.h>
#include <stdarg.h>
#include <stdio.h>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#error NULL not defined
#endif

#ifndef WCHAR_MAX
#error WCHAR_MAX not defined
#endif

#ifndef WCHAR_MIN
#error WCHAR_MIN not defined
#endif

#ifndef WEOF
#error WEOF not defined
#endif

// mbstate_t comes from the underlying C library; it is defined (in C99) as:
//    a complete object type other than an array type that can hold the conversion
//    state information necessary to convert between sequences of multibyte
//    characters and wide characters
mbstate_t mb = {};
size_t s = 0;
tm *tm = 0;
wint_t w = 0;
::FILE* fp = 0;
::va_list va;
char* ns = 0;
wchar_t* ws = 0;
const wchar_t* cws = 0;
wchar_t** wsp = 0;

ASSERT_SAME_TYPE(int,                decltype(fwprintf(fp, L"")));
ASSERT_SAME_TYPE(int,                decltype(fwscanf(fp, L"")));
ASSERT_SAME_TYPE(int,                decltype(swprintf(ws, s, L"")));
ASSERT_SAME_TYPE(int,                decltype(swscanf(L"", L"")));
ASSERT_SAME_TYPE(int,                decltype(vfwprintf(fp, L"", va)));
ASSERT_SAME_TYPE(int,                decltype(vfwscanf(fp, L"", va)));
ASSERT_SAME_TYPE(int,                decltype(vswprintf(ws, s, L"", va)));
ASSERT_SAME_TYPE(int,                decltype(vswscanf(L"", L"", va)));
ASSERT_SAME_TYPE(wint_t,             decltype(fgetwc(fp)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(fgetws(ws, 0, fp)));
ASSERT_SAME_TYPE(wint_t,             decltype(fputwc(L' ', fp)));
ASSERT_SAME_TYPE(int,                decltype(fputws(L"", fp)));
ASSERT_SAME_TYPE(int,                decltype(fwide(fp, 0)));
ASSERT_SAME_TYPE(wint_t,             decltype(getwc(fp)));
ASSERT_SAME_TYPE(wint_t,             decltype(putwc(L' ', fp)));
ASSERT_SAME_TYPE(wint_t,             decltype(ungetwc(L' ', fp)));
ASSERT_SAME_TYPE(double,             decltype(wcstod(L"", wsp)));
ASSERT_SAME_TYPE(float,              decltype(wcstof(L"", wsp)));
ASSERT_SAME_TYPE(long double,        decltype(wcstold(L"", wsp)));
ASSERT_SAME_TYPE(long,               decltype(wcstol(L"", wsp, 0)));
ASSERT_SAME_TYPE(long long,          decltype(wcstoll(L"", wsp, 0)));
ASSERT_SAME_TYPE(unsigned long,      decltype(wcstoul(L"", wsp, 0)));
ASSERT_SAME_TYPE(unsigned long long, decltype(wcstoull(L"", wsp, 0)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcscpy(ws, L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcsncpy(ws, L"", s)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcscat(ws, L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcsncat(ws, L"", s)));
ASSERT_SAME_TYPE(int,                decltype(wcscmp(L"", L"")));
ASSERT_SAME_TYPE(int,                decltype(wcscoll(L"", L"")));
ASSERT_SAME_TYPE(int,                decltype(wcsncmp(L"", L"", s)));
ASSERT_SAME_TYPE(size_t,             decltype(wcsxfrm(ws, L"", s)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcschr(ws, L' ')));
ASSERT_SAME_TYPE(const wchar_t*,     decltype(wcschr(cws, L' ')));
ASSERT_SAME_TYPE(size_t,             decltype(wcscspn(L"", L"")));
ASSERT_SAME_TYPE(size_t,             decltype(wcslen(L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcspbrk(ws, L"")));
ASSERT_SAME_TYPE(const wchar_t*,     decltype(wcspbrk(cws, L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcsrchr(ws, L' ')));
ASSERT_SAME_TYPE(const wchar_t*,     decltype(wcsrchr(cws, L' ')));
ASSERT_SAME_TYPE(size_t,             decltype(wcsspn(L"", L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcsstr(ws, L"")));
ASSERT_SAME_TYPE(const wchar_t*,     decltype(wcsstr(cws, L"")));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wcstok(ws, L"", wsp)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wmemchr(ws, L' ', s)));
ASSERT_SAME_TYPE(const wchar_t*,     decltype(wmemchr(cws, L' ', s)));
ASSERT_SAME_TYPE(int,                decltype(wmemcmp(L"", L"", s)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wmemcpy(ws, L"", s)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wmemmove(ws, L"", s)));
ASSERT_SAME_TYPE(wchar_t*,           decltype(wmemset(ws, L' ', s)));
ASSERT_SAME_TYPE(size_t,             decltype(wcsftime(ws, s, L"", tm)));
ASSERT_SAME_TYPE(wint_t,             decltype(btowc(0)));
ASSERT_SAME_TYPE(int,                decltype(wctob(w)));
ASSERT_SAME_TYPE(int,                decltype(mbsinit(&mb)));
ASSERT_SAME_TYPE(size_t,             decltype(mbrlen("", s, &mb)));
ASSERT_SAME_TYPE(size_t,             decltype(mbrtowc(ws, "", s, &mb)));
ASSERT_SAME_TYPE(size_t,             decltype(wcrtomb(ns, L' ', &mb)));
ASSERT_SAME_TYPE(size_t,             decltype(mbsrtowcs(ws, (const char**)0, s, &mb)));
ASSERT_SAME_TYPE(size_t,             decltype(wcsrtombs(ns, (const wchar_t**)0, s, &mb)));
ASSERT_SAME_TYPE(wint_t,             decltype(getwchar()));
ASSERT_SAME_TYPE(int,                decltype(vwscanf(L"", va)));
ASSERT_SAME_TYPE(int,                decltype(wscanf(L"")));
ASSERT_SAME_TYPE(wint_t,             decltype(putwchar(L' ')));
ASSERT_SAME_TYPE(int,                decltype(vwprintf(L"", va)));
ASSERT_SAME_TYPE(int,                decltype(wprintf(L"")));
