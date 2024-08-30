//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// inplace_vector();

#include <inplace_vector>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"
#include "../../../NotConstructible.h"

template <class C>
constexpr void test0() {
  static_assert(noexcept(C{}) && noexcept(C()));
  static_assert(std::is_trivially_default_constructible_v<C> == (C::capacity() == 0));
  C c;
  assert(c.empty());
  C c1 = {};
  assert(c1.empty());
}

struct NonTrivial {
  NonTrivial();
};

constexpr bool tests() {
  {
    using V = std::inplace_vector<int, 0>;
    test0<V>();
    constexpr V _;
  }
  {
    using V = std::inplace_vector<int, 10>;
    test0<V>();
    constexpr V _;
  }

  {
    using V = std::inplace_vector<NotConstructible, 0>;
    test0<V>();
    constexpr V _;
  }
  if !consteval {
    test0<std::inplace_vector<NotConstructible, 10>>();
  }

  {
    using V = std::inplace_vector<NonTrivial, 0>;
    test0<V>();
    constexpr V _;
  }
  if !consteval {
    test0<std::inplace_vector<NonTrivial, 10>>();
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
