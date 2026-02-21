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

// constexpr const_pointer operator->() const noexcept;
// constexpr pointer operator->() noexcept;

#include <cassert>
#include <concepts>
#include <memory>
#include <utility>

constexpr bool test() {
  {
    std::indirect<int> i;

    std::same_as<std::indirect<int>::pointer> decltype(auto) _       = i.operator->();
    std::same_as<std::indirect<int>::const_pointer> decltype(auto) _ = std::as_const(i).operator->();

    static_assert(noexcept(i.operator->()));
    static_assert(noexcept(std::as_const(i).operator->()));
  }
  {
    struct Incomplete;
    (void)([](std::indirect<Incomplete>& i) {
      (void)(i.operator->());
      (void)(std::as_const(i).operator->());
    });
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
