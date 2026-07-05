//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <ostream>

#include <ostream>
#include <sstream>
#include <utility>

#include "test_macros.h"

void f() {
  std::ostringstream s;
#ifndef TEST_HAS_NO_CHAR8_T
  char8_t c8_s[]       = u8"test";
  const char8_t* c8_cs = u8"test";
#endif
  char16_t c16_s[]       = u"test";
  const char16_t* c16_cs = u"test";
  char32_t c32_s[]       = U"test";
  const char32_t* c32_cs = U"test";

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  wchar_t w_s[]       = L"test";
  const wchar_t* w_cs = L"test";
  s << wchar_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  s << w_s;       // expected-error {{overload resolution selected deleted operator '<<'}}
  s << w_cs;      // expected-error {{overload resolution selected deleted operator '<<'}}

  std::wostringstream sw;
#  ifndef TEST_HAS_NO_CHAR8_T
  sw << char8_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c8_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c8_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}
#  endif

  sw << char16_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c16_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c16_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << char32_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c32_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << c32_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}

#endif // TEST_HAS_NO_WIDE_CHARACTERS

#ifndef TEST_HAS_NO_CHAR8_T
  s << char8_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c8_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c8_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}
#endif
  s << char16_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c16_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c16_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}
  s << char32_t(); // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c32_s;      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << c32_cs;     // expected-error {{overload resolution selected deleted operator '<<'}}
}
