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

// constexpr allocator_type get_allocator() const noexcept;

#include <cassert>
#include <concepts>
#include <memory>

constexpr bool test() {
  {
    const std::indirect<int> i;

    std::same_as<std::allocator<int>> decltype(auto) _ = i.get_allocator();

    static_assert(noexcept(i.get_allocator()));
  }
  {
    struct Incomplete;
    (void)([](std::indirect<Incomplete>& i) { return i.get_allocator(); });
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
