//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Ensure that APIs which take a CharT* (and no size for it) are diagnosing passing a nullptr to them

#include <string>

#include "test_macros.h"

void func() {
  const char* const np = nullptr;
  std::string str1(np);                         // expected-warning {{null passed}}
  std::string str2(np, std::allocator<char>{}); // expected-warning {{null passed}}
  str2 = np;                                    // expected-warning {{null passed}}
  str2 += np;                                   // expected-warning {{null passed}}
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

#if TEST_STD_VER >= 20
  str2.starts_with(np); // expected-warning {{null passed}}
  str2.ends_with(np);   // expected-warning {{null passed}}
#endif
#if TEST_STD_VER >= 23
  str2.contains(np); // expected-warning {{null passed}}
#endif
}
