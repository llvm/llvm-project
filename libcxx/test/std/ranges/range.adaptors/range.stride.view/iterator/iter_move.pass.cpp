//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  friend constexpr range_rvalue_reference_t<_Base> iter_move(__iterator const& __it)
//         noexcept(noexcept(ranges::iter_move(__it.__current_)))

#include <ranges>

#include "../types.h"

template <typename T>
concept iter_moveable = requires(T&& t) { std::ranges::iter_move(t); };

constexpr bool test() {
  {
    int iter_move_counter(0);
    using View       = IterSwapRange<true, true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(&iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(noexcept(std::ranges::iter_move(svb)));

    [[maybe_unused]] auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
  }

  {
    int iter_move_counter(0);
    using View       = IterSwapRange<true, false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(&iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(!noexcept(std::ranges::iter_move(svb)));

    [[maybe_unused]] auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
