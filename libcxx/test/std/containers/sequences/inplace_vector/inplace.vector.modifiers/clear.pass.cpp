//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr void clear() noexcept;

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 4> c{1, 2, 3};
  ASSERT_NOEXCEPT(c.clear());
  ASSERT_SAME_TYPE(decltype(c.clear()), void);
  c.clear();
  assert(c.empty());
  assert(c.capacity() == 4);
  c.clear();
  assert(c.empty());

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
