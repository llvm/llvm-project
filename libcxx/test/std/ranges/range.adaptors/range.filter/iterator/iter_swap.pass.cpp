//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// friend constexpr void iter_swap(iterator const& x, iterator const& y)
//  noexcept(noexcept(ranges::iter_swap(x.current_, y.current_)))
//  requires(indirectly_swappable<iterator_t<V>>);

#include <ranges>

#include <array>
#include <cassert>
#include <iterator>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class It>
concept has_iter_swap = requires (It it) {
  std::ranges::iter_swap(it, it);
};

struct IsEven {
  constexpr bool operator()(int x) const { return x % 2 == 0; }
};

template <class Iterator, bool IsNoexcept>
constexpr void test() {
  using Sentinel = sentinel_wrapper<Iterator>;
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, IsEven>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  {
    std::array<int, 5> array{1, 2, 1, 4, 1};
    FilterView view = make_filter_view(array.data(), array.data() + array.size(), IsEven{});
    FilterIterator const it1 = view.begin();
    FilterIterator const it2 = std::ranges::next(view.begin());

    static_assert(std::is_same_v<decltype(iter_swap(it1, it2)), void>);
    static_assert(noexcept(iter_swap(it1, it2)) == IsNoexcept);

    assert(*it1 == 2 && *it2 == 4); // test the test
    iter_swap(it1, it2);
    assert(*it1 == 4);
    assert(*it2 == 2);
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>,           /* noexcept */ false>();
  test<cpp20_input_iterator<int*>,           /* noexcept */ false>();
  test<forward_iterator<int*>,               /* noexcept */ false>();
  test<bidirectional_iterator<int*>,         /* noexcept */ false>();
  test<random_access_iterator<int*>,         /* noexcept */ false>();
  test<contiguous_iterator<int*>,            /* noexcept */ false>();
  test<int*,                                 /* noexcept */ true>();
  test<NoexceptIterSwapInputIterator<true>,  /* noexcept */ true>();
  test<NoexceptIterSwapInputIterator<false>, /* noexcept */ false>();

  // Test that iter_swap requires the underlying iterator to be iter_swappable
  {
    using Iterator = int const*;
    using View = minimal_view<Iterator, Iterator>;
    using FilterView = std::ranges::filter_view<View, IsEven>;
    using FilterIterator = std::ranges::iterator_t<FilterView>;
    static_assert(!std::indirectly_swappable<Iterator>);
    static_assert(!has_iter_swap<FilterIterator>);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
