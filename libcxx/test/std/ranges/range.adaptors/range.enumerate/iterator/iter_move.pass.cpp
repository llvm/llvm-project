//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// class enumerate_view

// class enumerate_view::iterator

// friend constexpr auto iter_move(const iterator& i)
//   noexcept(noexcept(ranges::iter_move(i.current_)) &&
//             is_nothrow_move_constructible_v<range_rvalue_reference_t<Base>>);

#include <array>
#include <cassert>
#include <ranges>
#include <type_traits>
#include <utility>

#include "test_iterators.h"

#include "../types.h"

template <class Iterator, bool HasNoexceptIterMove>
constexpr void test() {
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  std::array array{0, 1, 2, 3, 4};

  View mv{Iterator(std::to_address(base(array.begin()))), Sentinel(Iterator(std::to_address(base(array.end()))))};
  EnumerateView ev{std::move(mv)};
  EnumerateIterator const it = ev.begin();

  auto&& result = iter_move(it);

  using DifferenceT = std::iter_difference_t<EnumerateIterator>;
  static_assert(std::is_same_v<decltype(result), std::tuple<DifferenceT, int&&>&&>);

  assert(get<0>(result) == 0);
  assert(&get<1>(result) == std::to_address(base(array.begin())));

  static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
}

constexpr bool tests() {
  // clang-format off
  test<cpp17_input_iterator<int*>,                /* noexcept */ false>();
  test<cpp20_input_iterator<int*>,                /* noexcept */ false>();
  test<forward_iterator<int*>,                    /* noexcept */ false>();
  test<bidirectional_iterator<int*>,              /* noexcept */ false>();
  test<random_access_iterator<int*>,              /* noexcept */ false>();
  test<contiguous_iterator<int*>,                 /* noexcept */ false>();
  test<int*,                                      /* noexcept */ true>();
  test<MaybeNoexceptIterMoveInputIterator<true>,  /* noexcept */ true>();
  test<MaybeNoexceptIterMoveInputIterator<false>, /* noexcept */ false>();
  // clang-format on

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
