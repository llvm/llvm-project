//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// std::views::zip

#include <ranges>

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>
#include <utility>

#include "types.h"

static_assert(std::is_invocable_v<decltype((std::views::zip))>);
static_assert(!std::is_invocable_v<decltype((std::views::zip)), int>);
static_assert(std::is_invocable_v<decltype((std::views::zip)), SizedRandomAccessView>);
static_assert(
    std::is_invocable_v<decltype((std::views::zip)), SizedRandomAccessView, std::ranges::iota_view<int, int>>);
static_assert(!std::is_invocable_v<decltype((std::views::zip)), SizedRandomAccessView, int>);

constexpr bool test() {
  {
    // zip zero arguments
    auto v = std::views::zip();
    assert(std::ranges::empty(v));
    static_assert(std::is_same_v<decltype(v), std::ranges::empty_view<std::tuple<>>>);
  }

  {
    // zip a view
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::same_as<std::ranges::zip_view<SizedRandomAccessView>> decltype(auto) v =
        std::views::zip(SizedRandomAccessView{buffer});
    assert(std::ranges::size(v) == 8);
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, std::tuple<int&>>);
  }

  {
    // zip a viewable range
    std::array a{1, 2, 3};
    std::same_as<std::ranges::zip_view<std::ranges::ref_view<std::array<int, 3>>>> decltype(auto) v =
        std::views::zip(a);
    assert(&(std::get<0>(*v.begin())) == &(a[0]));
    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v)>, std::tuple<int&>>);
  }

  {
    // zip the zip_view
    int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::same_as<std::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>> decltype(auto) v =
        std::views::zip(SizedRandomAccessView{buffer}, SizedRandomAccessView{buffer});

    std::same_as<
        std::ranges::zip_view<std::ranges::zip_view<SizedRandomAccessView, SizedRandomAccessView>>> decltype(auto) v2 =
        std::views::zip(v);

    static_assert(std::is_same_v<std::ranges::range_reference_t<decltype(v2)>, std::tuple<std::pair<int&, int&>>>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
