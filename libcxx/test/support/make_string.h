//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_TEST_MAKE_STRING_H
#define SUPPORT_TEST_MAKE_STRING_H

#include "test_macros.h"

#if TEST_STD_VER < 11
#error This header requires C++11 or greater
#endif

#include <string>

#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
#define CHAR8_ONLY(x) x,
#else
#define CHAR8_ONLY(x)
#endif

#define MKSTR(Str)                                                             \
  {                                                                            \
    Str, TEST_CONCAT(L, Str),                                                  \
        CHAR8_ONLY(TEST_CONCAT(u8, Str)) TEST_CONCAT(u, Str),                  \
        TEST_CONCAT(U, Str)                                                    \
  }

struct MultiStringType {
  const char* s;
  const wchar_t* w;
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
  const char8_t* u8;
#endif
  const char16_t* u16;
  const char32_t* u32;

  constexpr operator const char*() const { return s; }
  constexpr operator const wchar_t*() const { return w; }
#if TEST_STD_VER > 17 && defined(__cpp_char8_t)
  constexpr operator const char8_t*() const { return u8; }
#endif
  constexpr operator const char16_t*() const { return u16; }
  constexpr operator const char32_t*() const { return u32; }
};

// Helper to convert a const char* string to a basic_string<CharT>.
// This helper is used in unit tests to make them generic. The input should be
// valid ASCII which means the input is also valid UTF-8.
#define MAKE_STRING(CharT, Str)                                                \
  std::basic_string<CharT> {                                                   \
    static_cast<const CharT*>(MultiStringType MKSTR(Str))                      \
  }

// Like MAKE_STRING but converts to a const CharT*.
#define MAKE_CSTRING(CharT, Str)                                               \
  static_cast<const CharT*>(MultiStringType MKSTR(Str))

#endif
