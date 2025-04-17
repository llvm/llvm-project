//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr iterator<false> end()       requires ((!simple-view <First> || ... || !simple-view <Vs>) && cartesian-product-is-common<      First,       Vs...>);
// constexpr iterator<true > end() const requires                                                        cartesian-product-is-common<const First, const Vs...> ;

#include "assert_macros.h"

#include <array>
#include <cassert>
#include <ranges>

constexpr bool test() {
  { // non-empty range
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{a};
    assert(c.end() == c.begin() + N);
  }

  { // (non-empty range)^2
    constexpr size_t N0 = 7;
    std::array<int, N0> a0;
    constexpr size_t N1 = 42;
    std::array<int, N1> a1;
    std::ranges::cartesian_product_view c{a0, a1};
    assert(c.end() == c.begin() + N0 * N1);
  }

  { // (non-empty range)^3 
    constexpr size_t N0 = 5, N1 = 42, N2 = 7;
    std::array<int, N0> a0;
    std::array<int, N1> a1;
    std::array<int, N2> a2;
    std::ranges::cartesian_product_view c{a0, a1, a2};
    assert(c.end() == c.begin() + N0*N1*N2);
  }

  { // empty range
    std::ranges::empty_view<int> e;
    std::ranges::cartesian_product_view c{e};
    assert(c.end() == c.begin());
  }

  { // (empty range)^2
    std::ranges::empty_view<int> e;
    std::ranges::cartesian_product_view c{e, e};
    assert(c.end() == c.begin());
  }

  { // empty range X common range
    std::ranges::empty_view<int> e;
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{e, a};
    assert(c.end() == c.begin());
  }

  { // common range X empty range
    std::ranges::empty_view<int> e;
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{e, a};
    assert(c.end() == c.begin());
  }

  { // (empty range)^3
    std::ranges::empty_view<int> e;
    std::ranges::cartesian_product_view c{e, e, e};
    assert(c.end() == c.begin());
  }

  { // empty range X empty range X common range
    std::ranges::empty_view<int> e;
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{e, e, a};
    assert(c.end() == c.begin());
  }

  { // empty range X common range X empty range 
    std::ranges::empty_view<int> e;
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{e, a, e};
    assert(c.end() == c.begin());
  }

  { // common range X empty range X empty range 
    std::ranges::empty_view<int> e;
    constexpr size_t N = 7;
    std::array<int, N> a;
    std::ranges::cartesian_product_view c{a, e, e};
    assert(c.end() == c.begin());
  }

  { // empty range X common range X common range 
    std::ranges::empty_view<int> e;
    constexpr size_t N0 = 7, N1 = 42;
    std::array<int, N0> a0;
    std::array<int, N1> a1;
    std::ranges::cartesian_product_view c{e, a0, a1};
    assert(c.end() == c.begin());
  }

  { // common range X empty range X common range 
    std::ranges::empty_view<int> e;
    constexpr size_t N0 = 7, N1 = 42;
    std::array<int, N0> a0;
    std::array<int, N1> a1;
    std::ranges::cartesian_product_view c{a0, e, a1};
    assert(c.end() == c.begin());
  }

  { // common range X common range X empty range 
    std::ranges::empty_view<int> e;
    constexpr size_t N0 = 7, N1 = 42;
    std::array<int, N0> a0;
    std::array<int, N1> a1;
    std::ranges::cartesian_product_view c{a0, a1, e};
    assert(c.end() == c.begin());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
