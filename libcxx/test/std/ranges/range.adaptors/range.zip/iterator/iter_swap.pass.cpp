//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr void iter_swap(const iterator& l, const iterator& r) noexcept(see below)
//   requires (indirectly_­swappable<iterator_t<maybe-const<Const, Views>>> && ...);

#include <array>
#include <cassert>
#include <ranges>

#include "../types.h"

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&){};
  ThrowingMove& operator=(ThrowingMove&&){return *this;}
};

constexpr bool test() {
  {
    std::array a1{1, 2, 3, 4};
    std::array a2{0.1, 0.2, 0.3};
    std::ranges::zip_view v(a1, a2);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();

    std::ranges::iter_swap(iter1, iter2);

    assert(a1[0] == 2);
    assert(a1[1] == 1);
    assert(a2[0] == 0.2);
    assert(a2[1] == 0.1);

    auto [x1, y1] = *iter1;
    assert(&x1 == &a1[0]);
    assert(&y1 == &a2[0]);

    auto [x2, y2] = *iter2;
    assert(&x2 == &a1[1]);
    assert(&y2 == &a2[1]);

    static_assert(noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // underlying iter_swap may throw
    std::array<ThrowingMove, 2> iterSwapMayThrow{};
    std::ranges::zip_view v(iterSwapMayThrow);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();
    static_assert(!noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // underlying iterators' iter_move are called through ranges::iter_swap
    adltest::IterMoveSwapRange r1, r2;
    assert(r1.iter_swap_called_times == 0);
    assert(r2.iter_swap_called_times == 0);

    std::ranges::zip_view v{r1, r2};
    auto it1 = v.begin();
    auto it2 = std::ranges::next(it1, 3);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 2);
    assert(r2.iter_swap_called_times == 2);

    std::ranges::iter_swap(it1, it2);
    assert(r1.iter_swap_called_times == 4);
    assert(r2.iter_swap_called_times == 4);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
