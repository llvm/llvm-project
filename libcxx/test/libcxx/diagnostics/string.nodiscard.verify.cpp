//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <string> functions are marked [[nodiscard]]

#include <string>
#include <string_view>

#include "test_macros.h"

std::string prval();

void test() {
  std::string str;
  const std::string cstr;
  std::string_view sv;

  str[0];     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr[0];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.at(0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.c_str(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.data();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.data(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.substr(0);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  prval().substr(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.front();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.back();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.begin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.begin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.end();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.end();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.rbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  cstr.rend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.cbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.cend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.capacity();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.empty();         // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.length();        // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.size();          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.max_size();      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.get_allocator(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.find(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.rfind(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.rfind(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.rfind(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.rfind("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.rfind("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // clang-format off
  str.find_first_of(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_of(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_of(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_of("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_of("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.find_last_of(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_of(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_of(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_of("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_of("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.find_first_not_of(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_not_of(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_not_of(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_not_of("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_first_not_of("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.find_last_not_of(str);      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_not_of(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_not_of(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_not_of("");       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.find_last_not_of("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.compare(str);          // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(sv);           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, sv);     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, sv, 0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, str);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, str, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare("");           // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, "");     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.compare(0, 0, "", 0);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  // clang-format on

#if TEST_STD_VER >= 20
  str.starts_with(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.starts_with(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.starts_with("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  str.ends_with(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.ends_with(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.ends_with("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 23
  str.contains(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.contains(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  str.contains("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 26
  str.subview(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}
