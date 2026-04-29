//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  friend constexpr range_rvalue_reference_t<Base> iter_move(const iterator& i)
//         noexcept(noexcept(ranges::iter_move(i.current_)))

#include <cassert>
#include <ranges>
#include <vector>

#include "../types.h"
#include "test_macros.h"

template <typename T>
concept iter_moveable = requires(T&& t) { std::ranges::iter_move(t); };

constexpr bool test() {
  {
    // iter_move with a noexcept iter_move on the base iterator.
    int a[] = {4, 3, 2, 1};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, /*IsSwappable=*/true, /*IsNoExcept=*/true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(svb)));
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
    // iter_move with a potentially-throwing iter_move on the base iterator.
    int a[] = {1, 2, 3, 4};

    int iter_move_counter(0);
    using View       = IterMoveIterSwapTestRange<int*, /*IsSwappable=*/true, /*IsNoExcept=*/false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svb         = StrideView(View(a, a + 4, &iter_move_counter), 1).begin();

    static_assert(iter_moveable<std::ranges::iterator_t<StrideView>>);
    ASSERT_SAME_TYPE(int, decltype(std::ranges::iter_move(svb)));
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
    // iter_move with a non-pointer base iterator (vector::iterator).
    std::vector<int> a = {4, 5, 6, 7, 8};

    int iter_move_counter(0);
    using View = IterMoveIterSwapTestRange<std::vector<int>::iterator, /*IsSwappable=*/true, /*IsNoExcept=*/false>;

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

  {
    // Verify return type is a rvalue-reference.
    int a[]    = {1, 2, 3};
    using Base = BasicTestView<int*, int*>;
    auto sv    = std::ranges::stride_view(Base(a, a + 3), 1);
    auto it    = sv.begin();
    static_assert(std::is_same_v<decltype(std::ranges::iter_move(it)), int&&>);
    assert(std::ranges::iter_move(it) == 1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
