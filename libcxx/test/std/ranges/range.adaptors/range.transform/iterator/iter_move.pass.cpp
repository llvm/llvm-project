//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr decltype(auto) iter_move(const iterator& i)
//    noexcept(noexcept(invoke(i.parent_->fun_, *i.current_)))

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    std::ranges::transform_view transformView(MoveOnlyView{buff}, PlusOneMutable{});
    auto iter = transformView.begin();
    ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(iter));

    assert(std::ranges::iter_move(iter) == 1);
    assert(std::ranges::iter_move(iter + 2) == 3);

    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(iter)));
    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(std::move(iter))));
  }

  {
    LIBCPP_ASSERT_NOEXCEPT(std::ranges::iter_move(
      std::declval<std::ranges::iterator_t<std::ranges::transform_view<MoveOnlyView, PlusOneNoexcept>>&>()));
    ASSERT_NOT_NOEXCEPT(std::ranges::iter_move(
      std::declval<std::ranges::iterator_t<std::ranges::transform_view<MoveOnlyView, PlusOneMutable>>&>()));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
