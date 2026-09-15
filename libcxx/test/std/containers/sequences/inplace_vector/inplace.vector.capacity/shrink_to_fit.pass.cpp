//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// static constexpr void shrink_to_fit() noexcept;

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

constexpr bool test() {
  using C = std::inplace_vector<int, 8>;
  C c{1, 2, 3};
  auto* data = c.data();
  c.shrink_to_fit();
  C::shrink_to_fit();
  assert(c.size() == 3);
  assert(c.capacity() == 8);
  assert(c.data() == data);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
