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
//     constexpr const iterator_t<V> base() const&;

#include <cassert>
#include <concepts>
#include <iterator>
#include <ranges>

#include "../types.h"

constexpr bool test() {
  int buffer[] = {1, 2, 3, 4};

  std::ranges::chunk_view<input_span<int>> chunked(input_span<int>(buffer, 4), 2);
  auto outer = chunked.begin();
  auto inner = (*outer).begin();

  std::same_as<const std::ranges::iterator_t<input_span<int>>> decltype(auto) base = inner.base();
  assert(*base == 1);

  ++inner;
  assert(*inner.base() == 2);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
