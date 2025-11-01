//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// friend constexpr range_rvalue_reference_t<_Base> iter_move(__iterator const& __it)
// noexcept(noexcept(ranges::iter_move(__it.__current_)))

#include <ranges>

#include "../../../../../std/ranges/range.adaptors/range.stride.view/types.h"

constexpr bool test() {
  {
    int a[] = {4, 3, 2, 1};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, true, true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(noexcept(std::ranges::iter_move(svb)));

    // These lines need to be in sync so that clang-verify knows where the warning comes from.
    // clang-format off
    std::ranges::iter_move( // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
        svb);
    // clang-format on
  }
  return true;
}
