//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cwchar>

// XFAIL: no-wide-characters

#include <cwchar>
#include <ctime>
#include <cstdarg>
#include <cstdio>
#include <type_traits>

#include "test_macros.h"

#ifndef NULL
#  error NULL not defined
#endif

#ifndef WCHAR_MAX
#  error WCHAR_MAX not defined
#endif

#ifndef WCHAR_MIN
#  error WCHAR_MIN not defined
#endif

#ifndef WEOF
#  error WEOF not defined
#endif

int main(int, char**) {
  std::mbstate_t mb = {};
  std::size_t s     = 0;
  std::tm* tm       = 0;
  std::wint_t w     = 0;
  ::FILE* fp        = 0;
  std::va_list va;

  char* ns           = 0;
  wchar_t* ws        = 0;
  const wchar_t* cws = 0;
  wchar_t** wsp      = 0;

  ((void)mb);  // Prevent unused warning
  ((void)s);   // Prevent unused warning
  ((void)tm);  // Prevent unused warning
  ((void)w);   // Prevent unused warning
  ((void)fp);  // Prevent unused warning
  ((void)va);  // Prevent unused warning
  ((void)ns);  // Prevent unused warning
  ((void)ws);  // Prevent unused warning
  ((void)cws); // Prevent unused warning
  ((void)wsp); // Prevent unused warning

  ASSERT_SAME_TYPE(int, decltype(std::fwprintf(fp, L"")));
  ASSERT_SAME_TYPE(int, decltype(std::fwscanf(fp, L"")));
  ASSERT_SAME_TYPE(int, decltype(std::swprintf(ws, s, L"")));
  ASSERT_SAME_TYPE(int, decltype(std::swscanf(L"", L"")));
  ASSERT_SAME_TYPE(int, decltype(std::vfwprintf(fp, L"", va)));
  ASSERT_SAME_TYPE(int, decltype(std::vfwscanf(fp, L"", va)));
  ASSERT_SAME_TYPE(int, decltype(std::vswprintf(ws, s, L"", va)));
  ASSERT_SAME_TYPE(int, decltype(std::vswscanf(L"", L"", va)));
  ASSERT_SAME_TYPE(std::wint_t, decltype(std::fgetwc(fp)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::fgetws(ws, 0, fp)));
  ASSERT_SAME_TYPE(std::wint_t, decltype(std::fputwc(L' ', fp)));
  ASSERT_SAME_TYPE(int, decltype(std::fputws(L"", fp)));
  ASSERT_SAME_TYPE(int, decltype(std::fwide(fp, 0)));
  ASSERT_SAME_TYPE(std::wint_t, decltype(std::getwc(fp)));
  ASSERT_SAME_TYPE(std::wint_t, decltype(std::putwc(L' ', fp)));
  ASSERT_SAME_TYPE(std::wint_t, decltype(std::ungetwc(L' ', fp)));
  ASSERT_SAME_TYPE(double, decltype(std::wcstod(L"", wsp)));
  ASSERT_SAME_TYPE(float, decltype(std::wcstof(L"", wsp)));
  ASSERT_SAME_TYPE(long double, decltype(std::wcstold(L"", wsp)));
  ASSERT_SAME_TYPE(long, decltype(std::wcstol(L"", wsp, 0)));
  ASSERT_SAME_TYPE(long long, decltype(std::wcstoll(L"", wsp, 0)));
  ASSERT_SAME_TYPE(unsigned long, decltype(std::wcstoul(L"", wsp, 0)));
  ASSERT_SAME_TYPE(unsigned long long, decltype(std::wcstoull(L"", wsp, 0)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcscpy(ws, L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcsncpy(ws, L"", s)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcscat(ws, L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcsncat(ws, L"", s)));
  ASSERT_SAME_TYPE(int, decltype(std::wcscmp(L"", L"")));
  ASSERT_SAME_TYPE(int, decltype(std::wcscoll(L"", L"")));
  ASSERT_SAME_TYPE(int, decltype(std::wcsncmp(L"", L"", s)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcsxfrm(ws, L"", s)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcschr(ws, L' ')));
  ASSERT_SAME_TYPE(const wchar_t*, decltype(std::wcschr(cws, L' ')));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcscspn(L"", L"")));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcslen(L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcspbrk(ws, L"")));
  ASSERT_SAME_TYPE(const wchar_t*, decltype(std::wcspbrk(cws, L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcsrchr(ws, L' ')));
  ASSERT_SAME_TYPE(const wchar_t*, decltype(std::wcsrchr(cws, L' ')));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcsspn(L"", L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcsstr(ws, L"")));
  ASSERT_SAME_TYPE(const wchar_t*, decltype(std::wcsstr(cws, L"")));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wcstok(ws, L"", wsp)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wmemchr(ws, L' ', s)));
  ASSERT_SAME_TYPE(const wchar_t*, decltype(std::wmemchr(cws, L' ', s)));
  ASSERT_SAME_TYPE(int, decltype(std::wmemcmp(L"", L"", s)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wmemcpy(ws, L"", s)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wmemmove(ws, L"", s)));
  ASSERT_SAME_TYPE(wchar_t*, decltype(std::wmemset(ws, L' ', s)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcsftime(ws, s, L"", tm)));
  ASSERT_SAME_TYPE(wint_t, decltype(std::btowc(0)));
  ASSERT_SAME_TYPE(int, decltype(std::wctob(w)));
  ASSERT_SAME_TYPE(int, decltype(std::mbsinit(&mb)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::mbrlen("", s, &mb)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::mbrtowc(ws, "", s, &mb)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcrtomb(ns, L' ', &mb)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::mbsrtowcs(ws, (const char**)0, s, &mb)));
  ASSERT_SAME_TYPE(std::size_t, decltype(std::wcsrtombs(ns, (const wchar_t**)0, s, &mb)));

  ASSERT_SAME_TYPE(std::wint_t, decltype(std::getwchar()));
  ASSERT_SAME_TYPE(int, decltype(std::vwscanf(L"", va)));
  ASSERT_SAME_TYPE(int, decltype(std::wscanf(L"")));

  ASSERT_SAME_TYPE(std::wint_t, decltype(std::putwchar(L' ')));
  ASSERT_SAME_TYPE(int, decltype(std::vwprintf(L"", va)));
  ASSERT_SAME_TYPE(int, decltype(std::wprintf(L"")));

  return 0;
}
