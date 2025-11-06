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

// constexpr const T& operator*() const & noexcept;
// constexpr T& operator*() & noexcept;

// constexpr const T&& operator*() const && noexcept;
// constexpr T&& operator*() && noexcept;

#include <cassert>
#include <concepts>
#include <memory>
#include <utility>

constexpr bool test() {
  {
    std::indirect<int> i;

    std::same_as<int&> decltype(auto) _        = *i;
    std::same_as<int&&> decltype(auto) _       = *std::move(i);
    std::same_as<const int&> decltype(auto) _  = *std::as_const(i);
    std::same_as<const int&&> decltype(auto) _ = *std::move(std::as_const(i));

    static_assert(noexcept(*i));
    static_assert(noexcept(*std::move(i)));
    static_assert(noexcept(*std::as_const(i)));
    static_assert(noexcept(*std::move(std::as_const(i))));
  }
  {
    struct Incomplete;
    (void)([](std::indirect<Incomplete>& i) {
      (void)(*i);
      (void)(*std::move(i));
      (void)(*std::as_const(i));
      (void)(*std::move(std::as_const(i)));
    });
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
