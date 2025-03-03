//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void reserve(); // Deprecated in C++20

// REQUIRES: c++20 || c++23

#include <string>

void f() {
  std::string s;
  s.reserve(); // expected-warning {{'reserve' is deprecated}}
}
