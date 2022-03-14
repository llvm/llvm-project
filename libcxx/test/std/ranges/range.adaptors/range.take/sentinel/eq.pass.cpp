//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// sentinel() = default;
// constexpr explicit sentinel(sentinel_t<Base> end);
// constexpr sentinel(sentinel<!Const> s)
//   requires Const && convertible_to<sentinel_t<V>, sentinel_t<Base>>;

#include <ranges>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "../types.h"

constexpr bool test() {
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    {
      const std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
      assert(tv.end() == std::ranges::next(tv.begin(), 4));
      assert(std::ranges::next(tv.begin(), 4) == tv.end());
    }

    {
      std::ranges::take_view<MoveOnlyView> tv(MoveOnlyView{buffer}, 4);
      assert(tv.end() == std::ranges::next(tv.begin(), 4));
      assert(std::ranges::next(tv.begin(), 4) == tv.end());
    }
  }

  {
    std::ranges::take_view<MoveOnlyView> tvNonConst(MoveOnlyView{buffer}, 4);
    const std::ranges::take_view<MoveOnlyView> tvConst(MoveOnlyView{buffer}, 4);
    assert(tvNonConst.end() == std::ranges::next(tvConst.begin(), 4));
    assert(std::ranges::next(tvConst.begin(), 4) == tvNonConst.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
