//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr decltype(auto) iter_move(const iterator& i);

#include <cassert>
#include <ranges>

#include "../types.h"

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    std::ranges::join_view jv(buffer);
    assert(std::ranges::iter_move(jv.begin()) == 1);
    static_assert(std::is_same_v<decltype(std::ranges::iter_move(jv.begin())), int&&>);

    static_assert(noexcept(std::ranges::iter_move(std::declval<decltype(jv.begin())>())));
  }

  {
    // iter_move calls inner's iter_move and calls
    // iter_move on the correct inner iterator
    IterMoveSwapAwareView inners[2] = {buffer[0], buffer[1]};
    std::ranges::join_view jv(inners);
    auto it = jv.begin();

    const auto& iter_move_called_times1 = jv.base().begin()->iter_move_called;
    const auto& iter_move_called_times2 = std::next(jv.base().begin())->iter_move_called;
    assert(iter_move_called_times1 == 0);
    assert(iter_move_called_times2 == 0);

    std::same_as<std::pair<int&&, int&&>> decltype(auto) x = std::ranges::iter_move(it);
    assert(std::get<0>(x) == 1);
    assert(iter_move_called_times1 == 1);
    assert(iter_move_called_times2 == 0);

    auto it2 = std::ranges::next(it, 4);

    std::same_as<std::pair<int&&, int&&>> decltype(auto) y = std::ranges::iter_move(it2);
    assert(std::get<0>(y) == 5);
    assert(iter_move_called_times1 == 1);
    assert(iter_move_called_times2 == 1);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
