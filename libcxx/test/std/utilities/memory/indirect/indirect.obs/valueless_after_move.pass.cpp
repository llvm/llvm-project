//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <memory>

// template <class T, class Allocator = std::allocator<T>> class indirect;

// constexpr bool valueless_after_move() const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

constexpr bool test() {
  {
    const std::indirect<int> i;

    std::same_as<bool> decltype(auto) _ = i.valueless_after_move();

    static_assert(noexcept(i.valueless_after_move()));
  }
  {
    struct Incomplete;
    (void)([](std::indirect<Incomplete>& i) { return i.valueless_after_move(); });
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
