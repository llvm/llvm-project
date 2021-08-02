//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// transform_view::<iterator>::operator[]

#include <ranges>

#include "test_macros.h"
#include "../types.h"

constexpr bool test() {
  int buff[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  {
    std::ranges::transform_view transformView(ContiguousView{buff}, Increment{});
    auto iter = transformView.begin();
    static_assert(!noexcept(std::ranges::iter_move(iter)));

    assert(std::ranges::iter_move(iter) == 1);
    assert(std::ranges::iter_move(iter + 2) == 3);

    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(iter)));
    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(std::move(iter))));
  }

  {
    static_assert( noexcept(std::ranges::iter_move(
      std::declval<std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, IncrementNoexcept>>&>())));
    static_assert(!noexcept(std::ranges::iter_move(
      std::declval<std::ranges::iterator_t<std::ranges::transform_view<ContiguousView, Increment>>&>())));
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
