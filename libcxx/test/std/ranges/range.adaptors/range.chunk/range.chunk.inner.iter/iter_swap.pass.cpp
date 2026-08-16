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
//     friend constexpr void iter_swap(const inner_iterator& x, const inner_iterator& y)
//       noexcept(noexcept(ranges::iter_swap(x.parent_->current_.value(), y.parent_->current_.value())))
//       requires indirectly_swappable<iterator_t<V>>;

#include <cassert>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  // Test `friend constexpr void iter_swap(const inner_iterator&, const inner_iterator&)`
  std::vector<int> vector_a = {1, 2, 3, 4};
  std::vector<int> vector_b = {5, 6, 7, 8};
  std::ranges::chunk_view<
      std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
      chunked_a(std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                    cpp17_input_iterator<int*>(vector_a.data()),
                    sentinel_wrapper<cpp17_input_iterator<int*>>(
                        cpp17_input_iterator<int*>(vector_a.data() + vector_a.size()))),
                2);
  std::ranges::chunk_view<
      std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
      chunked_b(std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
                    cpp17_input_iterator<int*>(vector_b.data()),
                    sentinel_wrapper<cpp17_input_iterator<int*>>(
                        cpp17_input_iterator<int*>(vector_b.data() + vector_b.size()))),
                2);

  auto inner_a = (*chunked_a.begin()).begin();
  auto inner_b = (*chunked_b.begin()).begin();

  assert(vector_a[0] == 1);
  assert(vector_b[0] == 5);

  std::ranges::iter_swap(inner_a, inner_b);

  assert(vector_a[0] == 5);
  assert(vector_b[0] == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
