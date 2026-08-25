//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// constexpr size_type size() const noexcept;

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 4> c;
  ASSERT_SAME_TYPE(std::inplace_vector<int, 4>::size_type, decltype(c.size()));
  assert(c.size() == 0);
  c.push_back(1);
  assert(c.size() == 1);
  c.push_back(2);
  assert(c.size() == 2);
  c.pop_back();
  assert(c.size() == 1);
  c.clear();
  assert(c.size() == 0);

  assert((std::inplace_vector<int, 0>().size() == 0));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
