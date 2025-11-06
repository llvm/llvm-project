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
#include <memory>
#include <utility>

constexpr bool test() {
  struct S {
    constexpr bool is_const() & noexcept { return false; }
    constexpr bool is_const() const& noexcept { return true; }
  };

  std::indirect<S> i;

  assert(!i->is_const());
  assert(std::as_const(i)->is_const());

  static_assert(noexcept(i->is_const()));
  static_assert(noexcept(std::as_const(i)->is_const()));

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
