//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Ensure that APIs which take a CharT* are diagnosing passing a nullptr to them

// Clang 19 and AppleClang don't have diagnose_if with diagnostic flags
// UNSUPPORTED: clang-19, apple-clang-17

#include <string_view>

#include "test_macros.h"

void func() {
  const char* const np = nullptr;
  std::string_view str1(np);  // expected-warning {{null passed}}
  str1 = np;                  // expected-warning {{null passed}}
  str1.find(np);              // expected-warning {{null passed}}
  str1.rfind(np);             // expected-warning {{null passed}}
  str1.find_first_of(np);     // expected-warning {{null passed}}
  str1.find_last_of(np);      // expected-warning {{null passed}}
  str1.find_first_not_of(np); // expected-warning {{null passed}}
  str1.find_last_not_of(np);  // expected-warning {{null passed}}
  str1.compare(np);           // expected-warning {{null passed}}
  str1.compare(0, 0, np);     // expected-warning {{null passed}}
  (void)(str1 == np);         // expected-warning {{null passed}}

#if TEST_STD_VER >= 20
  str1.starts_with(np); // expected-warning {{null passed}}
  str1.ends_with(np);   // expected-warning {{null passed}}
#endif
#if TEST_STD_VER >= 23
  str1.contains(np); // expected-warning {{null passed}}
#endif

  // clang-format off
  // These diagnostics are issued via diagnose_if, so we want to check the full description
  std::string_view str2(nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if len is not zero}}
  str2.find(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.rfind(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.find_first_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.find_last_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.find_first_not_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.find_last_not_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str2.compare(0, 0, nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n2 is not zero}}
  // clang-format on
}
