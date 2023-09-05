//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr value_type operator*() const;

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <functional>
#include <utility>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void test() {
  using Underlying      = View<Iter, Sent>;
  using ChunkByView     = std::ranges::chunk_by_view<Underlying, std::ranges::less_equal>;
  using ChunkByIterator = std::ranges::iterator_t<ChunkByView>;

  std::array array{0, 1, 2, 3, -1, 0, 1, 2, -2, 3, 4, 5};
  std::array expected{std::array{0, 1, 2, 3}, std::array{-1, 0, 1, 2}, std::array{-2, 3, 4, 5}};
  Underlying underlying{Iter{array.begin()}, Sent{Iter{array.end()}}};
  ChunkByView view{underlying, std::ranges::less_equal{}};

  size_t idx = 0;
  for (std::same_as<ChunkByIterator> auto iter = view.begin(); iter != view.end(); ++idx, ++iter) {
    std::same_as<typename ChunkByIterator::value_type> auto chunk = *iter;
    assert(std::ranges::equal(chunk, expected[idx]));
  }
}

constexpr bool tests() {
  // Check iterator-sentinel pair
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  // Check iterator pair
  test<forward_iterator<int*>, forward_iterator<int*>>();
  test<bidirectional_iterator<int*>, bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>, random_access_iterator<int*>>();
  test<contiguous_iterator<int*>, contiguous_iterator<int*>>();
  test<int*, int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
