//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++23

// <ranges>

//   V models forward_range:
//     constexpr iterator_t<Base> base() const;

#include <cassert>
#include <concepts>
#include <ranges>
#include <vector>

#include "test_iterators.h"

constexpr bool test() {
  // Test `constexpr iterator_t<Base> base() const`
  {
    static_assert(std::same_as< decltype(std::declval<std::ranges::iterator_t< std::ranges::chunk_view<
                                             std::ranges::subrange<forward_iterator<int*>,
                                                                   sentinel_wrapper<forward_iterator<int*>>>>> const&>()
                                             .base()),
                                forward_iterator<int*>>);

    std::vector<int> vector = {1, 2, 3, 4, 5, 6};
    std::ranges::chunk_view<std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>>
        chunked(std::ranges::subrange<forward_iterator<int*>, sentinel_wrapper<forward_iterator<int*>>>(
                    forward_iterator<int*>(vector.data()),
                    sentinel_wrapper<forward_iterator<int*>>(forward_iterator<int*>(vector.data() + vector.size()))),
                2);
    auto it = chunked.begin();

    assert(base(it.base()) == vector.data());
    ++it;
    assert(base(it.base()) == vector.data() + 2);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
