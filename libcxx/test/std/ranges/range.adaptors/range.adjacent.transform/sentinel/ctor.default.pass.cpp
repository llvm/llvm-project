//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// sentinel() = default;

#include <cassert>
#include <ranges>
#include <tuple>

#include "../helpers.h"

struct PODSentinel {
  bool b; // deliberately uninitialised

  friend constexpr bool operator==(int*, const PODSentinel& s) { return s.b; }
};

struct Range : std::ranges::view_base {
  int* begin() const;
  PODSentinel end();
};

template <std::size_t N, class Fn>
constexpr void test() {
  {
    using R        = std::ranges::adjacent_transform_view<Range, Fn, N>;
    using Sentinel = std::ranges::sentinel_t<R>;
    static_assert(!std::is_same_v<Sentinel, std::ranges::iterator_t<R>>);

    std::ranges::iterator_t<R> it;

    Sentinel s1;
    assert(it != s1); // PODSentinel.b is initialised to false

    Sentinel s2 = {};
    assert(it != s2); // PODSentinel.b is initialised to false
  }
}

template <std::size_t N>
constexpr void test() {
  test<N, MakeTuple>();
  test<N, Tie>();
  test<N, GetFirst>();
  test<N, Multiply>();
}

constexpr bool test() {
  test<1>();
  test<2>();
  test<3>();
  test<4>();
  test<5>();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
