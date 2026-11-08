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

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>
#include <utility>

#include "test_iterators.h"
#include "test_range.h"

constexpr bool test() {
  // Test `friend constexpr range_rvalue_reference_t<V> iter_move(const inner_iterator&)`
  static_assert(std::same_as<decltype(std::ranges::iter_move(
                                 std::declval<const std::ranges::iterator_t< std::ranges::range_reference_t<
                                     std::ranges::chunk_view<test_view<cpp20_input_iterator>>>>&>())),
                             int&&>);
  std::vector<int> vector = {1, 2, 3, 4, 5, 6, 7, 8};
  std::ranges::chunk_view<
      std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
      chunked =
          std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
              cpp17_input_iterator<int*>(vector.data()),
              sentinel_wrapper<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))) |
          std::views::chunk(2);
  assert(std::ranges::iter_move((*chunked.begin()).begin()) == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
