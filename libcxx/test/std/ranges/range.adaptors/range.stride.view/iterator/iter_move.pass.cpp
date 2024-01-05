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
#include <vector>

#include "../types.h"
#include "__ranges/access.h"

template <typename T>
concept iter_moveable = requires(T&& t) { std::ranges::iter_move(t); };

constexpr bool test() {
  {
    int a[] = {4, 3, 2, 1};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, true, true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(noexcept(std::ranges::iter_move(svb)));

    auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
    assert(result == 4);

    svb++;
    result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 2);
    assert(result == 3);
  }

  {
    int a[] = {1, 2, 3, 4};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, true, false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    static_assert(std::is_same_v<int, decltype(std::ranges::iter_move(svb))>);
    static_assert(!noexcept(std::ranges::iter_move(svb)));

    auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
    assert(result == 1);

    svb++;
    result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 2);
    assert(result == 2);
  }

  {
    std::vector<int> a = {4, 5, 6, 7, 8};

    int iter_move_counter(0);
    using View = IterMoveIterSwapTestRange<std::vector<int>::iterator, true, false>;

    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a.begin(), a.end(), &iter_move_counter), 1).begin();

    static_assert(!noexcept(std::ranges::iter_move(svb)));

    auto&& result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 1);
    assert(result == 4);

    svb++;
    result = std::ranges::iter_move(svb);
    assert(iter_move_counter == 2);
    assert(result == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
