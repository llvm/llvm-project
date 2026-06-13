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

// template<class T, class Allocator>
// struct hash<indirect<T, Allocator>>;

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_convertible.h"

constexpr bool test() {
  { // Hashing an indirect hashes its owned object.
    std::indirect<int> i1(1);
    assert(std::hash<std::indirect<int>>()(i1) == std::hash<int>()(1));
  }
  { // Hashing a valueless indirect is valid and returns an implementation-defined value.
    std::indirect<int> i1(1);
    std::indirect<int> i2(2);
    auto(std::move(i1));
    auto(std::move(i2));
    assert(std::hash<std::indirect<int>>()(i1) == std::hash<std::indirect<int>>()(i2));
  }
  { // hash<indirect<T>> is only enabled if hash<T> is.
    static_assert(std::is_default_constructible_v<std::hash<std::indirect<int>>>);
    struct S {};
    static_assert(!std::is_default_constructible_v<std::hash<std::indirect<S>>>);
  }
  return true;
}

int main(int, char**) {
  test();
  return 0;
}
