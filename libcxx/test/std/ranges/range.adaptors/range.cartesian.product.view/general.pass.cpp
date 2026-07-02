//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// Broad smoke test exercising cartesian_product_view across several range categories,
// pipelining, and use with standard algorithms.

#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <ranges>
#include <tuple>

constexpr bool test() {
  { // 2-range product: full enumeration in lexicographic order
    std::array a{1, 2, 3};
    std::array b{'a', 'b'};
    int idx                          = 0;
    std::tuple<int, char> expected[] = {{1, 'a'}, {1, 'b'}, {2, 'a'}, {2, 'b'}, {3, 'a'}, {3, 'b'}};
    for (auto [x, y] : std::views::cartesian_product(a, b)) {
      auto t = std::tuple{x, y};
      assert(t == expected[idx]);
      ++idx;
    }
    assert(idx == 6);
  }

  { // empty middle range collapses the product
    std::array a{1, 2, 3};
    std::array<int, 0> e{};
    std::array c{1.0, 2.0};
    auto v   = std::views::cartesian_product(a, e, c);
    int seen = 0;
    for (auto _ : v)
      ++seen;
    assert(seen == 0);
    assert(v.empty());
  }

  { // chained with std::views::transform -- works as a typical pipeline
    std::array a{1, 2};
    std::array b{10, 20, 30};
    auto sums = std::views::cartesian_product(a, b) |
                std::views::transform([](auto t) { return std::get<0>(t) + std::get<1>(t); });
    int total = std::ranges::fold_left(sums, 0, std::plus{});
    // (1+10)+(1+20)+(1+30)+(2+10)+(2+20)+(2+30) = 11+21+31+12+22+32 = 129
    assert(total == 129);
  }

  { // random-access iterator arithmetic on the cartesian iterator matches manual indexing
    std::array a{1, 2, 3, 4};
    std::array b{10, 20};
    auto v  = std::views::cartesian_product(a, b);
    auto it = v.begin();
    assert(it[0] == std::tuple(1, 10));
    assert(it[1] == std::tuple(1, 20));
    assert(it[2] == std::tuple(2, 10));
    assert(it[7] == std::tuple(4, 20));
    assert(v.end() - it == 8);
  }

  { // size() agrees with the product of range sizes for a 4-range product
    std::array a{1};
    std::array b{1, 2};
    std::array c{1, 2, 3};
    std::array d{1, 2, 3, 4};
    auto v = std::views::cartesian_product(a, b, c, d);
    assert(v.size() == 1u * 2u * 3u * 4u);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
