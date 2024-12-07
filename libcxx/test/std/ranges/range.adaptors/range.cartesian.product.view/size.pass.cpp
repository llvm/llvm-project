//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::cartesian_product_view::size

#include <array>
#include <cassert>
#include <initializer_list>
#include <ranges>

constexpr bool test() {
  { // example taken from: https://en.cppreference.com/w/cpp/ranges/cartesian_product_view/size
    constexpr static auto w = {1};
    constexpr static auto x = {2, 3};
    constexpr static auto y = {4, 5, 6};
    constexpr static auto z = {7, 8, 9, 10, 11, 12, 13};

    constexpr auto ww = std::ranges::views::all(w);
    constexpr auto xx = std::ranges::views::all(x);
    constexpr auto yy = std::ranges::views::all(y);
    constexpr auto zz = std::ranges::views::all(z);

    constexpr auto v = std::ranges::cartesian_product_view(ww, xx, yy, zz);

    assert(v.size() == 42);
    assert(v.size() == w.size() * x.size() * y.size() * z.size());
  }

  { // empty range
    std::ranges::empty_view<int> e;
    auto v = std::ranges::cartesian_product_view(e);
    assert(v.size() == 0);
  }

  { // 1..3 range(s)
    constexpr size_t N0 = 3, N1 = 7, N2 = 42;
    std::array<int, N0> a0;
    std::array<int, N1> a1;
    std::array<int, N2> a2;
    assert(std::ranges::cartesian_product_view(a0).size() == N0);
    assert(std::ranges::cartesian_product_view(a0, a1).size() == N0 * N1);
    assert(std::ranges::cartesian_product_view(a0, a1, a2).size() == N0 * N1 * N2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}