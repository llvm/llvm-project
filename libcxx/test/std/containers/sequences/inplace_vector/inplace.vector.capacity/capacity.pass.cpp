//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// static constexpr size_type capacity() noexcept;

#include <cassert>
#include <inplace_vector>

#include "test_macros.h"

constexpr bool test() {
  {
    using C = std::inplace_vector<int, 0>;
    ASSERT_SAME_TYPE(C::size_type, decltype(C::capacity()));
    C c;
    assert(c.capacity() == 0);
    assert(C::capacity() == 0);
  }
  {
    using C = std::inplace_vector<int, 42>;
    ASSERT_SAME_TYPE(C::size_type, decltype(C::capacity()));
    C c{1, 2, 3};
    assert(c.capacity() == 42);
    assert(C::capacity() == 42);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
