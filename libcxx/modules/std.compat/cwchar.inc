// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export {
#ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using ::mbstate_t;
  using ::size_t;
  using ::wint_t;

  using ::tm;

  using ::btowc;
  using ::fgetwc;
  using ::fgetws;
  using ::fputwc;
  using ::fputws;
  using ::fwide;
  using ::fwprintf;
  using ::fwscanf;
  using ::getwc;
  using ::getwchar;
  using ::putwc;
  using ::putwchar;
  using ::swprintf;
  using ::swscanf;
  using ::ungetwc;
  using ::vfwprintf;
  using ::vfwscanf;
  using ::vswprintf;
  using ::vswscanf;
  using ::vwprintf;
  using ::vwscanf;
  using ::wcscat;
  using ::wcschr;
  using ::wcscmp;
  using ::wcscoll;
  using ::wcscpy;
  using ::wcscspn;
  using ::wcsftime;
  using ::wcslen;
  using ::wcsncat;
  using ::wcsncmp;
  using ::wcsncpy;
  using ::wcspbrk;
  using ::wcsrchr;
  using ::wcsspn;
  using ::wcsstr;
  using ::wcstod;
  using ::wcstof;
  using ::wcstok;
  using ::wcstol;
  using ::wcstold;
  using ::wcstoll;
  using ::wcstoul;
  using ::wcstoull;
  using ::wcsxfrm;
  using ::wctob;
  using ::wmemchr;
  using ::wmemcmp;
  using ::wmemcpy;
  using ::wmemmove;
  using ::wmemset;
  using ::wprintf;
  using ::wscanf;

  // [c.mb.wcs], multibyte / wide string and character conversion functions
  using ::mbrlen;
  using ::mbrtowc;
  using ::mbsinit;
  using ::mbsrtowcs;
  using ::wcrtomb;
  using ::wcsrtombs;
#endif // _LIBCPP_HAS_NO_WIDE_CHARACTERS
} // export
