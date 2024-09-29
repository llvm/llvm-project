//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// void clear() noexcept;

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    std::inplace_vector<int, 10> c{1, 2, 3};
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
  }
  {
    std::inplace_vector<int, 0> c;
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
  }
  {
    std::inplace_vector<MoveOnly, 0> c;
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
  }
  if !consteval {
    std::inplace_vector<MoveOnly, 10> c;
    c.push_back(0);
    ASSERT_NOEXCEPT(c.clear());
    c.clear();
    assert(c.empty());
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
