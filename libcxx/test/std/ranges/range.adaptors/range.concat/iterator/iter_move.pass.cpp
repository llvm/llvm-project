//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <cassert>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

struct Range : std::ranges::view_base {
  using Iterator = forward_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Iterator begin() const { return Iterator(begin_); }
  constexpr Sentinel end() const { return Sentinel(Iterator(end_)); }

private:
  int* begin_;
  int* end_;
};

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
};

template <class Iterator, bool HasNoexceptIterMove>
constexpr bool test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View     = minimal_view<Iterator, Sentinel>;

  {
    std::array<int, 5> array1{0, 1, 2, 3, 4};
    std::array<int, 5> array2{5, 6, 7, 8, 9};

    View v1{Iterator(array1.data()), Sentinel(Iterator(array1.data() + array1.size()))};
    View v2{Iterator(array2.data()), Sentinel(Iterator(array2.data() + array2.size()))};
    std::ranges::concat_view view(std::move(v1), std::move(v2));

    auto it = view.begin();
    assert(std::ranges::iter_move(view.begin()) == 0);
    static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
  }

  {
    // iter_move may throw
    auto throwingMoveRange =
        std::views::iota(0, 2) | std::views::transform([](auto) noexcept { return ThrowingMove{}; });
    std::ranges::concat_view v(throwingMoveRange);
    auto it = v.begin();
    static_assert(!noexcept(std::ranges::iter_move(it)));
  }

  return true;
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>, /* noexcept */ false>();
  test<forward_iterator<int*>, /* noexcept */ false>();
  test<bidirectional_iterator<int*>, /* noexcept */ false>();
  test<random_access_iterator<int*>, /* noexcept */ false>();
  test<contiguous_iterator<int*>, /* noexcept */ false>();
  test<int*, /* noexcept */ true>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
