//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// Some basic examples of how elements_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <algorithm>
#include <array>
#include <cassert>
#include <map>
#include <ranges>
#include <string_view>
#include <tuple>
#include <utility>

#include "test_iterators.h"

int main(int, char**) {
  using namespace std::string_view_literals;
  auto historicalFigures =
      std::map{std::pair{"Lovelace"sv, 1815}, {"Turing"sv, 1912}, {"Babbage"sv, 1791}, {"Hamilton"sv, 1936}};
  auto expectedYears = {1791, 1936, 1815, 1912};

  // views::elements<N>
  {
    auto names         = historicalFigures | std::views::elements<0>;
    auto expectedNames = {"Babbage"sv, "Hamilton"sv, "Lovelace"sv, "Turing"sv};
    assert(std::ranges::equal(names, expectedNames));

    auto birth_years = historicalFigures | std::views::elements<1>;
    assert(std::ranges::equal(birth_years, expectedYears));
  }

  // views::keys
  {
    auto names         = historicalFigures | std::views::keys;
    auto expectedNames = {"Babbage"sv, "Hamilton"sv, "Lovelace"sv, "Turing"sv};
    assert(std::ranges::equal(names, expectedNames));
  }

  // views::values
  {
    auto is_even = [](const auto x) { return x % 2 == 0; };
    assert(std::ranges::count_if(historicalFigures | std::views::values, is_even) == 2);
  }

  // array
  {
    std::array<int, 3> arrs[] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto ev                   = arrs | std::views::elements<2>;
    auto expected             = {3, 6, 9};
    assert(std::ranges::equal(ev, expected));
  }

  // pair
  {
    std::pair<double, int> ps[] = {{1.0, 2}, {3.0, 4}, {5.0, 6}};
    auto ev                     = ps | std::views::elements<1>;
    auto expected               = {2, 4, 6};
    assert(std::ranges::equal(ev, expected));
  }

  // tuple
  {
    std::tuple<short> tps[] = {{short{1}}, {short{2}}, {short{3}}};
    auto ev                 = tps | std::views::elements<0>;
    auto expected           = {1, 2, 3};
    assert(std::ranges::equal(ev, expected));
  }

  // subrange
  {
    int is[]       = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    using Iter     = forward_iterator<int*>;
    using Sent     = sentinel_wrapper<Iter>;
    using SubRange = std::ranges::subrange<Iter, Sent>;
    SubRange sr[]  = {
        {Iter{is}, Sent{Iter{is + 1}}},
        {Iter{is + 2}, Sent{Iter{is + 5}}},
        {Iter{is + 6}, Sent{Iter{is + 8}}},
    };

    auto iters = sr | std::views::elements<0>;
    static_assert(std::is_same_v<Iter, std::ranges::range_reference_t<decltype(iters)>>);
    auto expectedIters = {is, is + 2, is + 6};
    assert(std::ranges::equal(iters | std::views::transform([](auto&& iter) { return base(iter); }), expectedIters));

    auto sentinels = sr | std::views::elements<1>;
    static_assert(std::is_same_v<Sent, std::ranges::range_reference_t<decltype(sentinels)>>);
    auto expectedSentinels = {is + 1, is + 5, is + 8};
    assert(std::ranges::equal(
        sentinels | std::views::transform([](auto&& st) { return base(base(st)); }), expectedSentinels));
  }
  return 0;
}
