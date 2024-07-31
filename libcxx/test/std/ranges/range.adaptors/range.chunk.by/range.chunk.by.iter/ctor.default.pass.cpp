//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// std::ranges::chunk_by_view<V>::<iterator>() = default;

#include <ranges>

#include <cassert>
#include <functional>
#include <type_traits>
#include <utility>

#include "../types.h"
#include "test_iterators.h"

template <class Iterator, bool IsNoexcept>
constexpr void testDefaultConstructible() {
  // Make sure the iterator is default constructible.
  using ChunkByView     = std::ranges::chunk_by_view<View<Iterator>, std::ranges::less_equal>;
  using ChunkByIterator = std::ranges::iterator_t<ChunkByView>;
  ChunkByIterator i{};
  ChunkByIterator j;
  assert(i == j);
  static_assert(noexcept(ChunkByIterator{}) == IsNoexcept);
}

constexpr bool tests() {
  testDefaultConstructible<forward_iterator<int*>, /*IsNoexcept=*/false>();
  testDefaultConstructible<bidirectional_iterator<int*>, /*IsNoexcept=*/false>();
  testDefaultConstructible<random_access_iterator<int*>, /*IsNoexcept=*/false>();
  testDefaultConstructible<contiguous_iterator<int*>, /*IsNoexcept=*/false>();
  testDefaultConstructible<int*, /*IsNoexcept=*/true>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
