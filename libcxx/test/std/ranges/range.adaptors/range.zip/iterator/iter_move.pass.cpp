//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr auto iter_move(const iterator& i) noexcept(see below);

#include <array>
#include <cassert>
#include <iterator>
#include <ranges>
#include <tuple>

#include "../types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
};

constexpr bool test() {
  {
    // underlying iter_move noexcept
    std::array a1{1, 2, 3, 4};
    const std::array a2{3.0, 4.0};

    std::ranges::zip_view v(a1, a2, std::views::iota(3L));
    assert(std::ranges::iter_move(v.begin()) == std::make_tuple(1, 3.0, 3L));
    static_assert(std::is_same_v<decltype(std::ranges::iter_move(v.begin())), std::tuple<int&&, const double&&, long>>);

    auto it = v.begin();
    static_assert(noexcept(std::ranges::iter_move(it)));
  }

  {
    // underlying iter_move may throw
    auto throwingMoveRange =
        std::views::iota(0, 2) | std::views::transform([](auto) noexcept { return ThrowingMove{}; });
    std::ranges::zip_view v(throwingMoveRange);
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_move
    adltest::IterMoveSwapRange r1{}, r2{};
    assert(r1.iter_move_called_times == 0);
    assert(r2.iter_move_called_times == 0);
    std::ranges::zip_view v(r1, r2);
    auto it = v.begin();
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
      assert(r1.iter_move_called_times == 1);
      assert(r2.iter_move_called_times == 1);
    }
    {
      [[maybe_unused]] auto&& i = std::ranges::iter_move(it);
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
