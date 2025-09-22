//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <string>

//    constexpr basic_string_view<charT, traits> subview(size_type pos = 0,
//                                                       size_type n = npos) const;

#include <string>

void test() {
  std::string s;

  s.subview(); // expected-warning {{ignoring return value of function}}
}
