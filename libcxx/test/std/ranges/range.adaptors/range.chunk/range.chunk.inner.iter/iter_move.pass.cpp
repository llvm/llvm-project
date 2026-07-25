//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models only input_range:
//     friend constexpr range_rvalue_reference_t<V> iter_move(const inner_iterator& i)
//       noexcept(noexcept(ranges::iter_move(i.parent_->current_.value())));

#include <concepts>
#include <ranges>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

constexpr bool test() {
  using InnerIterator =
      std::ranges::iterator_t<std::ranges::range_reference_t<std::ranges::chunk_view<test_view<cpp20_input_iterator>>>>;

  static_assert(std::ranges::input_range<test_view<cpp20_input_iterator>>);
  static_assert(!std::ranges::forward_range<test_view<cpp20_input_iterator>>);

  static_assert(std::same_as<decltype(std::ranges::iter_move(std::declval<const InnerIterator&>())), int&&>);
  static_assert(std::same_as<decltype(std::ranges::iter_move(std::declval<const InnerIterator&>())),
                             std::ranges::range_rvalue_reference_t<test_view<cpp20_input_iterator>>>);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}