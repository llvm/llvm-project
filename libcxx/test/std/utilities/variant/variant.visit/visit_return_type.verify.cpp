//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

#include <variant>

// Verify that binding returned reference to local temporary object is rejected
// (per P2255R2 in library, P2748R5 in core langauge).

void test() {
  // expected-error@variant:* {{returning reference to local temporary object}}
  std::visit<const int&>([](char c) { return c; }, std::variant<char>{'*'});
}
