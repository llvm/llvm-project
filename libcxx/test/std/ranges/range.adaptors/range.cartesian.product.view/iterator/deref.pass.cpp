//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// constexpr auto operator*() const;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>
#include <utility>

#include "../../range_adaptor_types.h"

constexpr bool test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};

  { // single range
    std::ranges::cartesian_product_view v(a);
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&>>);
  }

  { // operator* is const-qualified
    std::ranges::cartesian_product_view v(a);
    const auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&>>);
  }

  { // two heterogeneous ranges -- references are to underlying elements
    std::ranges::cartesian_product_view v(a, b);
    auto it     = v.begin();
    auto [x, y] = *it;
    assert(&x == &(a[0]));
    assert(&y == &(b[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, double&>>);

    // Verify mutability through the reference tuple.
    x = 5;
    y = 0.1;
    assert(a[0] == 5);
    assert(b[0] == 0.1);
  }

  { // prvalue range_reference_t (views::iota)
    std::ranges::cartesian_product_view v(a, b, std::views::iota(0, 5));
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(b[0]));
    assert(std::get<2>(*it) == 0);
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, double&, int>>);
  }

  { // const-correctness -- const lvalue range becomes const reference
    std::ranges::cartesian_product_view v(a, std::as_const(a));
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(a[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, int const&>>);
  }

  { // dereference at end-1 returns the last element of the rightmost range
    std::ranges::cartesian_product_view v(a, b);
    auto last   = v.begin() + (a.size() * b.size() - 1);
    auto [x, y] = *last;
    assert(&x == &(a[a.size() - 1]));
    assert(&y == &(b[b.size() - 1]));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
