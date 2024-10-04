//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::ranges::cartesian_product_view::size

#include <cassert>
#include <ranges>
#include <initializer_list>

constexpr bool test() {
  { // testing: constexpr auto size() const
    // example taken from: https://en.cppreference.com/w/cpp/ranges/cartesian_product_view/size
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

  return true;
}

int main() {
  test();
  static_assert(test());
}
