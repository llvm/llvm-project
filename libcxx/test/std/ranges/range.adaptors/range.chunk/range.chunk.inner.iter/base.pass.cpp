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
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  // Test `constexpr const iterator_t<V> base() const&`
  std::vector<int> vector = {1, 2, 3, 4};
  std::ranges::chunk_view<
      std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>>
      chunked(
          std::ranges::subrange<cpp17_input_iterator<int*>, sentinel_wrapper<cpp17_input_iterator<int*>>>(
              cpp17_input_iterator<int*>(vector.data()),
              sentinel_wrapper<cpp17_input_iterator<int*>>(cpp17_input_iterator<int*>(vector.data() + vector.size()))),
          2);
  auto outer = chunked.begin();
  auto inner = (*outer).begin();

  std::same_as<const cpp17_input_iterator<int*>> decltype(auto) base = inner.base();
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
