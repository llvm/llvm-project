//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

#include <ranges>

#include <array>
#include <cassert>
#include <iterator>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class It>
concept has_iter_swap = requires(It it) { std::ranges::iter_swap(it, it); };

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) {}
  ThrowingMove& operator=(ThrowingMove&&) { return *this; }
};

template <class Iterator, bool IsNoexcept>
constexpr void test() {
  using Sentinel   = sentinel_wrapper<Iterator>;
  using View       = minimal_view<Iterator, Sentinel>;
  using ConcatView = std::ranges::concat_view<View>;

  {
    std::array<int, 5> array1{0, 1, 2, 3, 4};
    std::array<int, 5> array2{5, 6, 7, 8, 9};

    View v1{Iterator(array1.data()), Sentinel(Iterator(array1.data() + array1.size()))};
    View v2{Iterator(array2.data()), Sentinel(Iterator(array2.data() + array2.size()))};
    std::ranges::concat_view view(std::move(v1), std::move(v2));

    auto it1 = view.begin();
    auto it2 = ++view.begin();

    static_assert(std::is_same_v<decltype(iter_swap(it1, it2)), void>);
    static_assert(noexcept(iter_swap(it1, it2)) == IsNoexcept);

    assert(*it1 == 0 && *it2 == 1);
    iter_swap(it1, it2);
    assert(*it1 == 1);
    assert(*it2 == 0);
  }

  {
    // iter swap may throw
    std::array<ThrowingMove, 2> iterSwapMayThrow{};
    std::ranges::concat_view v(iterSwapMayThrow);
    auto iter1 = v.begin();
    auto iter2 = ++v.begin();
    static_assert(!noexcept(std::ranges::iter_swap(iter1, iter2)));
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>, /* noexcept */ false>();
  test<forward_iterator<int*>, /* noexcept */ false>();
  test<bidirectional_iterator<int*>, /* noexcept */ false>();
  test<random_access_iterator<int*>, /* noexcept */ false>();
  test<contiguous_iterator<int*>, /* noexcept */ false>();
  test<int*, /* noexcept */ false>();

  // Test that iter_swap requires the underlying iterator to be iter_swappable
  {
    using Iterator       = int const*;
    using View           = minimal_view<Iterator, Iterator>;
    using ConcatView     = std::ranges::concat_view<View>;
    using ConcatIterator = std::ranges::iterator_t<ConcatView>;
    static_assert(!std::indirectly_swappable<Iterator>);
    static_assert(!has_iter_swap<ConcatIterator>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
