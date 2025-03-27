//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// constexpr auto operator*() const;

#include <array>
#include <cassert>
#include <ranges>
#include <tuple>

#include "../types.h"

constexpr bool test() {
  std::array a{1, 2, 3, 4};
  std::array b{4.1, 3.2, 4.3};
  {
    // single range
    std::ranges::zip_view v(a);
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&>>);
  }

  {
    // operator* is const
    std::ranges::zip_view v(a);
    const auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
  }

  {
    // two ranges with different types
    std::ranges::zip_view v(a, b);
    auto it = v.begin();
    auto [x, y] = *it;
    assert(&x == &(a[0]));
    assert(&y == &(b[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, double&>>);

    x = 5;
    y = 0.1;
    assert(a[0] == 5);
    assert(b[0] == 0.1);
  }

  {
    // underlying range with prvalue range_reference_t
    std::ranges::zip_view v(a, b, std::views::iota(0, 5));
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(b[0]));
    assert(std::get<2>(*it) == 0);
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, double&, int>>);
  }

  {
    // const-correctness
    std::ranges::zip_view v(a, std::as_const(a));
    auto it = v.begin();
    assert(&(std::get<0>(*it)) == &(a[0]));
    assert(&(std::get<1>(*it)) == &(a[0]));
    static_assert(std::is_same_v<decltype(*it), std::tuple<int&, int const&>>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
