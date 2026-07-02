//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// friend constexpr auto iter_move(const iterator& i) noexcept(...);

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <tuple>

#include "../../range_adaptor_types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
};

constexpr bool test() {
  { // underlying iter_move is noexcept on common ranges
    std::array a{1, 2, 3, 4};
    const std::array b{3.0, 4.0};
    std::ranges::cartesian_product_view v(a, b, std::views::iota(3L, 6L));
    auto it = v.begin();

    assert(std::ranges::iter_move(it) == std::make_tuple(1, 3.0, 3L));
    static_assert(std::is_same_v<decltype(std::ranges::iter_move(it)), std::tuple<int&&, const double&&, long>>);
    static_assert(noexcept(std::ranges::iter_move(it)));
  }

  { // underlying iter_move can throw -> cartesian iter_move is not noexcept
    auto throwingRange = std::views::iota(0, 2) | std::views::transform([](auto) noexcept { return ThrowingMove{}; });
    std::ranges::cartesian_product_view v(throwingRange);
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));
  }

  { // ADL-customised iter_move on the underlying iterator is invoked
    adltest::IterMoveSwapRange r1{}, r2{};
    assert(r1.iter_move_called_times == 0);
    assert(r2.iter_move_called_times == 0);
    std::ranges::cartesian_product_view v(r1, r2);
    auto it = v.begin();
    {
      [[maybe_unused]] auto&& moved = std::ranges::iter_move(it);
      assert(r1.iter_move_called_times == 1);
      assert(r2.iter_move_called_times == 1);
    }
    {
      [[maybe_unused]] auto&& moved = std::ranges::iter_move(it);
      assert(r1.iter_move_called_times == 2);
      assert(r2.iter_move_called_times == 2);
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
