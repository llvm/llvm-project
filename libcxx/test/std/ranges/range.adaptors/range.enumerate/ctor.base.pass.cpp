//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// constexpr explicit enumerate_view(V base);

#include <ranges>

#include <array>
#include <cassert>
#include <tuple>
#include <type_traits>

#include "types.h"

constexpr bool test() {
  using EnumerateView = std::ranges::enumerate_view<RangeView>;

  {
    std::array base = {0, 1, 2, 3, 84};

    RangeView range(base.begin(), base.end());
    EnumerateView view{range};

    auto baseIt = base.begin();
    auto viewIt = view.begin();
    for (std::size_t index = 0; index != base.size(); ++index) {
      auto [vi, vv] = *viewIt;
      assert(std::cmp_equal(index, vi));
      assert(*baseIt == vv);

      ++baseIt;
      ++viewIt;
    }
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
