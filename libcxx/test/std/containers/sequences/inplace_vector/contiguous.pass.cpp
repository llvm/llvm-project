//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

#include <cassert>
#include <inplace_vector>
#include <memory>

constexpr bool test() {
  {
    std::inplace_vector<int, 8> c{1, 2, 3};
    assert(std::to_address(c.begin()) == c.data());
    assert(std::to_address(c.begin() + 1) == c.data() + 1);
    assert(std::to_address(c.end()) == c.data() + c.size());
  }
  {
    const std::inplace_vector<int, 8> c{1, 2, 3};
    assert(std::to_address(c.begin()) == c.data());
    assert(std::to_address(c.begin() + 1) == c.data() + 1);
    assert(std::to_address(c.end()) == c.data() + c.size());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
