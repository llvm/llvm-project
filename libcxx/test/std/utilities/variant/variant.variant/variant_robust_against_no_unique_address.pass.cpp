//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++17

// <variant>

#include <cassert>
#include <variant>

#include "test_macros.h"

struct S {
  TEST_NO_UNIQUE_ADDRESS std::variant<int, void*> a;
  bool b;
};

TEST_CONSTEXPR_CXX20 bool test() {
  S x{{}, true};
  x.a.emplace<0>();
  x.a.emplace<1>();
  return x.b;
}

int main() {
  assert(test());
#if TEST_STD_VER >= 20
  static_assert(test());
#endif
}
