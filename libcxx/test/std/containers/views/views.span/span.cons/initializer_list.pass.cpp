//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <span>

// constexpr explicit(extent != dynamic_extent) span(std::initializer_list<value_type> il); // Since C++26

#include <array>
#include <cassert>
#include <cstddef>
#include <initializer_list>
#include <span>

#include "test_macros.h"

struct Sink {
  constexpr Sink() = default;
  constexpr Sink(Sink*) {}
};

constexpr std::size_t count(std::span<const Sink> sp) { return sp.size(); }

template <std::size_t N>
constexpr std::size_t count_n(std::span<const Sink, N> sp) {
  return sp.size();
}

constexpr bool test() {
#if TEST_STD_VER >= 26
  // Dynamic extent
  {
    Sink a[10];

    assert(count({a}) == 1);
    assert(count({a, a + 10}) == 2);
    assert(count({a, a + 1, a + 2}) == 3);
    assert(count(std::initializer_list<Sink>{a[0], a[1], a[2], a[3]}) == 4);
  }
#else
  {
    Sink a[10];

    assert(count({a}) == 10);
    assert(count({a, a + 10}) == 10);
    assert(count_n<10>({a}) == 10);
  }
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
