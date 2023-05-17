//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr iterator_t<V> const& base() const& noexcept;
// constexpr iterator_t<V> base() &&;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  std::array<int, 5> array{0, 1, 2, 3, 4};
  FilterView view = make_filter_view(array.begin(), array.end(), AlwaysTrue{});

  // Test the const& version
  {
    FilterIterator const iter(view, Iterator(array.begin()));
    Iterator const& result = iter.base();
    ASSERT_SAME_TYPE(Iterator const&, decltype(iter.base()));
    ASSERT_NOEXCEPT(iter.base());
    assert(base(result) == array.begin());
  }

  // Test the && version
  {
    FilterIterator iter(view, Iterator(array.begin()));
    Iterator result = std::move(iter).base();
    ASSERT_SAME_TYPE(Iterator, decltype(std::move(iter).base()));
    assert(base(result) == array.begin());
  }
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  test<int const*>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
