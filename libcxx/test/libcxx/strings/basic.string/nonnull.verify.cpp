//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Ensure that APIs which take a CharT* are diagnosing passing a nullptr to them

// AppleClang doesn't have diagnose_if with diagnostic flags
// UNSUPPORTED: apple-clang-17

// ADDITIONAL_COMPILE_FLAGS: -Wno-unused-result

#include <string>

#include "test_macros.h"

void func() {
  const char* const np = nullptr;
  std::string str1(np);                         // expected-warning {{null passed}}
  std::string str2(np, std::allocator<char>{}); // expected-warning {{null passed}}
  str2 = np;                                    // expected-warning {{null passed}}
  str2 += np;                                   // expected-warning {{null passed}}
  str2.assign(np);                              // expected-warning {{null passed}}
  str2.append(np);                              // expected-warning {{null passed}}
  str2.insert(0, np);                           // expected-warning {{null passed}}
  str2.find(np);                                // expected-warning {{null passed}}
  str2.rfind(np);                               // expected-warning {{null passed}}
  str2.find_first_of(np);                       // expected-warning {{null passed}}
  str2.find_last_of(np);                        // expected-warning {{null passed}}
  str2.find_first_not_of(np);                   // expected-warning {{null passed}}
  str2.find_last_not_of(np);                    // expected-warning {{null passed}}
  str2.compare(np);                             // expected-warning {{null passed}}
  str2.compare(0, 0, np);                       // expected-warning {{null passed}}
  str2.replace(0, 0, np);                       // expected-warning {{null passed}}
  (void)(str2 == np);                           // expected-warning {{null passed}}

#if TEST_STD_VER >= 20
  str2.starts_with(np); // expected-warning {{null passed}}
  str2.ends_with(np);   // expected-warning {{null passed}}
#endif
#if TEST_STD_VER >= 23
  str2.contains(np); // expected-warning {{null passed}}
#endif

  // clang-format off
  // These diagnostics are issued via diagnose_if, so we want to check the full description
  std::string str3(nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  std::string str4(nullptr, 1, std::allocator<char>{}); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.find(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.rfind(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.find_first_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.find_last_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.find_first_not_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.find_last_not_of(nullptr, 0, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.compare(0, 0, nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n2 is not zero}}
  str4.assign(nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.append(nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.insert(0, nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n is not zero}}
  str4.replace(0, 0, nullptr, 1); // expected-warning {{null passed to callee that requires a non-null argument if n2 is not zero}}
  // clang-format on
}
