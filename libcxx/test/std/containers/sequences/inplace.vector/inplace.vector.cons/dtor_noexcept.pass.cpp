//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// ~inplace_vector() // implied noexcept;

#include <inplace_vector>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

constexpr bool tests() {
  {
    using C = std::inplace_vector<int, 0>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
  {
    using C = std::inplace_vector<int, 10>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
  {
    using C = std::inplace_vector<MoveOnly, 0>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }
  {
    using C = std::inplace_vector<MoveOnly, 10>;
    static_assert(std::is_nothrow_destructible_v<C>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
