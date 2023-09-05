// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::mbstate_t;
  using std::size_t;
  using std::wint_t;

  using std::tm;

  using std::btowc;
  using std::fgetwc;
  using std::fgetws;
  using std::fputwc;
  using std::fputws;
  using std::fwide;
  using std::fwprintf;
  using std::fwscanf;
  using std::getwc;
  using std::getwchar;
  using std::putwc;
  using std::putwchar;
  using std::swprintf;
  using std::swscanf;
  using std::ungetwc;
  using std::vfwprintf;
  using std::vfwscanf;
  using std::vswprintf;
  using std::vswscanf;
  using std::vwprintf;
  using std::vwscanf;
  using std::wcscat;
  using std::wcschr;
  using std::wcscmp;
  using std::wcscoll;
  using std::wcscpy;
  using std::wcscspn;
  using std::wcsftime;
  using std::wcslen;
  using std::wcsncat;
  using std::wcsncmp;
  using std::wcsncpy;
  using std::wcspbrk;
  using std::wcsrchr;
  using std::wcsspn;
  using std::wcsstr;
  using std::wcstod;
  using std::wcstof;
  using std::wcstok;
  using std::wcstol;
  using std::wcstold;
  using std::wcstoll;
  using std::wcstoul;
  using std::wcstoull;
  using std::wcsxfrm;
  using std::wctob;
  using std::wmemchr;
  using std::wmemcmp;
  using std::wmemcpy;
  using std::wmemmove;
  using std::wmemset;
  using std::wprintf;
  using std::wscanf;

  // [c.mb.wcs], multibyte / wide string and character conversion functions
  using std::mbrlen;
  using std::mbrtowc;
  using std::mbsinit;
  using std::mbsrtowcs;
  using std::wcrtomb;
  using std::wcsrtombs;
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS
} // namespace std
