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
#include <concepts>
#include <functional>
#include <ranges>
#include <string>
#include <tuple>
#include <vector>

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

  { // a std::views::transform base is non-simple: iterator_t<V> and iterator_t<const V> differ,
    // so traversing the product instantiates __iterator<false>'s __next. Helpers that hardcoded
    // the const base type failed to compile here. Both orders, since only one of them recursed
    // into the wrapping path.
    std::array a{1, 2};
    auto t = a | std::views::transform([](const auto& value) { return 2 * value + 1; });

    {
      auto v = std::views::cartesian_product(a, t);
      // Pin the specialisation: a simple base would silently drop this coverage.
      static_assert(!std::same_as<decltype(v.begin()), std::ranges::iterator_t<const decltype(v)>>);
      int seen = 0;
      for (auto _ : v)
        ++seen;
      assert(seen == 4);
    }
    {
      auto v = std::views::cartesian_product(t, a);
      static_assert(!std::same_as<decltype(v.begin()), std::ranges::iterator_t<const decltype(v)>>);
      int seen = 0;
      for (auto _ : v)
        ++seen;
      assert(seen == 4);
    }
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

  { // [range.cartesian.overview] 25.7.33.1.3 Example 1
    struct ConstexprStringStream {
      std::string str;

      constexpr ConstexprStringStream& operator<<(int x) { return *this << char(x + 48); }
      constexpr ConstexprStringStream& operator<<(char c) {
        str += c;
        return *this;
      }
    };

    const std::vector<int> v{0, 1, 2};
    ConstexprStringStream out;
    for (auto&& [a, b, c] : std::ranges::views::cartesian_product(v, v, v)) {
      out << a << ' ' << b << ' ' << c << '\n';
    }

    const std::string_view expected =
        "0 0 0\n"
        "0 0 1\n"
        "0 0 2\n"
        "0 1 0\n"
        "0 1 1\n"
        "0 1 2\n"
        "0 2 0\n"
        "0 2 1\n"
        "0 2 2\n"
        "1 0 0\n"
        "1 0 1\n"
        "1 0 2\n"
        "1 1 0\n"
        "1 1 1\n"
        "1 1 2\n"
        "1 2 0\n"
        "1 2 1\n"
        "1 2 2\n"
        "2 0 0\n"
        "2 0 1\n"
        "2 0 2\n"
        "2 1 0\n"
        "2 1 1\n"
        "2 1 2\n"
        "2 2 0\n"
        "2 2 1\n"
        "2 2 2\n";
    assert(out.str == expected);
  }

  // LWG3801: "cartesian_product_view::iterator::distance-from ignores 
  // the size of last underlying range".
  { 
    int x[] = {1, 2, 3};
    auto v  = std::views::cartesian_product(x, x);
    auto i  = v.begin() + 5; // *i == {2, 3}
    assert((*i == std::tuple{2, 3}));
    assert(i - v.begin() == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
