//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

//  friend constexpr void iter_swap(const iterator& x, const iterator& y)
//  noexcept(noexcept(ranges::iter_swap(x.current_, y.current_)))
//  requires indirectly_swappable<iterator_t<Base>>

#include <cassert>
#include <ranges>

#include "../types.h"

template <typename T>
concept Swappable = requires(T&& t, T&& u) { std::ranges::iter_swap(t, u); };

constexpr bool test() {
  {
    // iter_swap with a noexcept iter_swap on the base iterator.
    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8};

    int iter_move_counter_one(0);
    int iter_move_counter_two(0);
    using View       = IterMoveIterSwapTestRange<int*, /*IsSwappable=*/true, /*IsNoExcept=*/true>;
    using StrideView = std::ranges::stride_view<View>;
    auto svba        = StrideView(View(a, a + 4, &iter_move_counter_one), 1).begin();
    auto svbb        = StrideView(View(b, b + 4, &iter_move_counter_two), 1).begin();

    static_assert(Swappable<std::ranges::iterator_t<StrideView>>);
    static_assert(noexcept(std::ranges::iter_swap(svba, svbb)));

    assert(a[0] == 1);
    assert(b[0] == 5);

    std::ranges::iter_swap(svba, svbb);
    assert(iter_move_counter_one == 1);
    assert(iter_move_counter_two == 1);

    assert(a[0] == 5);
    assert(b[0] == 1);
  }

  {
    // iter_swap with a potentially-throwing iter_swap on the base iterator.
    int a[] = {1, 2, 3, 4};
    int b[] = {5, 6, 7, 8};

    int iter_move_counter_one(0);
    int iter_move_counter_two(0);
    using View       = IterMoveIterSwapTestRange<int*, /*IsSwappable=*/true, /*IsNoExcept=*/false>;
    using StrideView = std::ranges::stride_view<View>;
    auto svba        = StrideView(View(a, a + 4, &iter_move_counter_one), 1).begin();
    auto svbb        = StrideView(View(b, b + 4, &iter_move_counter_two), 1).begin();

    static_assert(Swappable<std::ranges::iterator_t<StrideView>>);
    static_assert(!noexcept(std::ranges::iter_swap(svba, svbb)));

    assert(a[0] == 1);
    assert(b[0] == 5);

    std::ranges::iter_swap(svba, svbb);

    assert(iter_move_counter_one == 1);
    assert(iter_move_counter_two == 1);
    assert(a[0] == 5);
    assert(b[0] == 1);
  }

  {
    // iter_swap is not available when the base iterator is not indirectly_swappable.
    using View       = IterMoveIterSwapTestRange<int*, /*IsSwappable=*/false, /*IsNoExcept=*/false>;
    using StrideView = std::ranges::stride_view<View>;

    static_assert(!Swappable<std::ranges::iterator_t<StrideView>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
