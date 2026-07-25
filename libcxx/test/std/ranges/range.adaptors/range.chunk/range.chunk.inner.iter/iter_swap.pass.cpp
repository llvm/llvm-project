//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

// V models only input_range:
//   friend constexpr void iter_swap(const inner_iterator& x, const inner_iterator& y)
//     noexcept(noexcept(ranges::iter_swap(x.parent_->current_.value(), y.parent_->current_.value())))
//     requires indirectly_swappable<iterator_t<V>>;

#include <cassert>
#include <ranges>

#include "../types.h"

constexpr bool test() {
  int a[] = {1, 2, 3, 4};
  int b[] = {5, 6, 7, 8};

  // Each chunk_view owns its own `current_` cache, so use two independent views to observe a real swap.
  std::ranges::chunk_view<input_span<int>> chunked_a(input_span<int>(a, 4), 2);
  std::ranges::chunk_view<input_span<int>> chunked_b(input_span<int>(b, 4), 2);

  auto inner_a = (*chunked_a.begin()).begin();
  auto inner_b = (*chunked_b.begin()).begin();

  assert(a[0] == 1);
  assert(b[0] == 5);

  std::ranges::iter_swap(inner_a, inner_b);

  assert(a[0] == 5);
  assert(b[0] == 1);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
