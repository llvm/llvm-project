//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// friend constexpr bool operator==(const iterator& x, const iterator& y);
// friend constexpr bool operator==(const iterator& x, default_sentinel_t);

#include <ranges>

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

  auto make_chunk_by_view = [](auto& arr) {
    View view{Iter(arr.data()), Sent(Iter(arr.data() + arr.size()))};
    return ChunkByView(std::move(view), std::ranges::less_equal{});
  };

  // Test operator==
  {
    std::array array{0, 1, 2};
    ChunkByView view  = make_chunk_by_view(array);
    ChunkByIterator i = view.begin();
    ChunkByIterator j = view.begin();

    std::same_as<bool> decltype(auto) result = (i == j);
    assert(result);
    ++i;
    assert(!(i == j));
  }

  // Test synthesized operator!=
  {
    std::array array{0, 1, 2};
    ChunkByView view  = make_chunk_by_view(array);
    ChunkByIterator i = view.begin();
    ChunkByIterator j = view.begin();

    std::same_as<bool> decltype(auto) result = (i != j);
    assert(!result);
    ++i;
    assert(i != j);
  }

  // Test operator== with std::default_sentinel_t
  {
    std::array array{0, 1, 2};
    ChunkByView view  = make_chunk_by_view(array);
    ChunkByIterator i = view.begin();

    std::same_as<bool> decltype(auto) result = (i == std::default_sentinel);
    assert(!result);
    ++i;
    assert(i == std::default_sentinel);
  }

  // Test synthesized operator!= with std::default_sentinel_t
  {
    std::array array{0, 1, 2};
    ChunkByView view  = make_chunk_by_view(array);
    ChunkByIterator i = view.begin();

    std::same_as<bool> decltype(auto) result = (i != std::default_sentinel);
    assert(result);
    ++i;
    assert(!(i != std::default_sentinel));
  }
}

struct TestWithPair {
  template <class Iterator>
  constexpr void operator()() const {
    // Test with pair of iterators
    test<Iterator, Iterator>();

    // Test with iterator-sentinel pair
    test<Iterator>();
  }
};

constexpr bool tests() {
  TestWithPair tester;
  types::for_each(types::forward_iterator_list<int*>{}, tester);
  types::for_each(types::forward_iterator_list<int const*>{}, tester);

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
