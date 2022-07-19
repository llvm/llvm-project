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

int main() {
  std::ostringstream s;

#ifndef TEST_HAS_NO_WIDE_CHARACTERS

  s << wchar_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<wchar_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<const wchar_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}

  std::wostringstream sw;
#  ifndef TEST_HAS_NO_CHAR8_T
  sw << char8_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<char8_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<const char8_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}
#  endif

  sw << char16_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<char16_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<const char16_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << char32_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<char32_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  sw << std::declval<const char32_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}

#endif // TEST_HAS_NO_WIDE_CHARACTERS

#ifndef TEST_HAS_NO_CHAR8_T
  s << char8_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<char8_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<const char8_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}
#endif
  s << char16_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<char16_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<const char16_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}
  s << char32_t();                      // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<char32_t*>();       // expected-error {{overload resolution selected deleted operator '<<'}}
  s << std::declval<const char32_t*>(); // expected-error {{overload resolution selected deleted operator '<<'}}
}
