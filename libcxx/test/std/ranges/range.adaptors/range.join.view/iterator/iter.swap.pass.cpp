//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// friend constexpr void iter_swap(const iterator& x, const iterator& y);

#include <cassert>
#include <ranges>

#include "../types.h"

using NonSwappableView = BufferView<copying_iterator<int*>>;
static_assert(std::ranges::input_range<NonSwappableView>);
static_assert(!std::indirectly_swappable<std::ranges::iterator_t<NonSwappableView>>);

constexpr bool test() {
  int buffer[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};

  {
    std::ranges::join_view jv(buffer);
    auto iter1 = jv.begin();
    auto iter2 = std::next(jv.begin());
    assert(buffer[0][0] == 1);
    assert(buffer[0][1] == 2);
    std::ranges::iter_swap(iter1, iter2);
    assert(buffer[0][0] == 2);
    assert(buffer[0][1] == 1);

    static_assert(noexcept(std::ranges::iter_swap(iter1, iter2)));
  }

  {
    // iter_move calls inner's iter_swap
    IterMoveSwapAwareView inners[1] = {buffer[0]};
    std::ranges::join_view jv(inners);
    auto it1 = jv.begin();
    auto it2 = std::ranges::next(it1);

    const auto& iter_swap_called_times = jv.base().begin()->iter_swap_called;

    assert(iter_swap_called_times == 0);
    assert(buffer[0][0] == 2);
    assert(buffer[0][1] == 1);

    std::ranges::iter_swap(it1, it2);

    assert(buffer[0][0] == 1);
    assert(buffer[0][1] == 2);
    assert(iter_swap_called_times == 1);
  }

  {
    // LWG3517 `join_view::iterator`'s `iter_swap` is underconstrained
    // `iter_swap` should not be defined if Inner's iterator does not indirectly_swappable
    NonSwappableView inners[2] = {buffer[0], buffer[1]};
    std::ranges::join_view jv(inners);
    using Iter = std::ranges::iterator_t<decltype(jv)>;
    static_assert(!std::indirectly_swappable<Iter>);
  }
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
