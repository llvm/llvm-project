//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-unicode

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// Ensure that APIs which take a FILE* are diagnosing passing a nullptr to them

#include <print>

void func() {
  std::print(nullptr, "");                                      // expected-warning {{null passed}}
  std::println(nullptr, "");                                    // expected-warning {{null passed}}
  std::println(nullptr);                                        // expected-warning {{null passed}}
  std::vprint_unicode(nullptr, "", std::make_format_args());    // expected-warning {{null passed}}
  std::vprint_nonunicode(nullptr, "", std::make_format_args()); // expected-warning {{null passed}}
}
