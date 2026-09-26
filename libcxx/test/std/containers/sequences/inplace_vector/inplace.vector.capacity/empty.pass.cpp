//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr bool empty() const noexcept;

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 4> c;
  ASSERT_SAME_TYPE(bool, decltype(c.empty()));
  assert(c.empty());
  c.push_back(1);
  assert(!c.empty());
  c.clear();
  assert(c.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
