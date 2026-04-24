//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// check that <string_view> functions are marked [[nodiscard]]

#include <string_view>

#include "type_algorithms.h"
#include "test_macros.h"

void test_members() {
  std::string_view sv;

  sv.begin();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.end();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.cbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.cend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.rbegin();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.rend();    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.crbegin(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.crend();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.size();     // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.length();   // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.max_size(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.empty(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv[0];    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.at(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.front(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.back();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.data();  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.substr(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#if TEST_STD_VER >= 26
  sv.subview(); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare(sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare(0, 0, sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare(0, 0, sv, 0, 0);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare("");
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare(0, 0, "");
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.compare(0, 0, "", 0);

  sv.find(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find("", 0);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.rfind(sv);       // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.rfind(' ');      // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.rfind("", 0, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.rfind("", 0);    // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_of(sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_of(' ');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_of("", 0, 0);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_of("", 0);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_of(sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_of(' ');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_of("", 0, 0);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_of("", 0);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_not_of(sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_not_of(' ');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_not_of("", 0, 0);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_first_not_of("", 0);

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_not_of(sv);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_not_of(' ');
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_not_of("", 0, 0);
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.find_last_not_of("", 0);

#if TEST_STD_VER >= 20
  sv.starts_with(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.starts_with(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.starts_with("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  sv.ends_with(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.ends_with(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.ends_with("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif

#if TEST_STD_VER >= 23
  sv.contains(sv);  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.contains(' '); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  sv.contains("");  // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
#endif
}

void test_nonmembers() {
  // std::hash<>

  std::hash<std::string_view> hash;
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  hash(std::string_view());

#if TEST_STD_VER >= 14
  // string_view literals

  using namespace std::string_view_literals;

  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  ""sv; // const char*
#  if !defined(TEST_HAS_NO_WIDE_CHARACTERS)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  L""sv; // const wchar_t*
#  endif
#  if !defined(TEST_HAS_NO_CHAR8_T)
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  u8""sv; // const char8_t*
#  endif
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  u""sv; // const char16_t*
  // expected-warning@+1 {{ignoring return value of function declared with 'nodiscard' attribute}}
  U""sv; // const char32_t*
#endif
}
