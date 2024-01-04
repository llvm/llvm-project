//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr range_reference_t<V> operator*() const;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <utility>
#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iterator, class ValueType = int, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  auto make_filter_view = [](auto begin, auto end, auto pred) {
    View view{Iterator(begin), Sentinel(Iterator(end))};
    return FilterView(std::move(view), pred);
  };

  std::array array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  FilterView view = make_filter_view(array.data(), array.data() + array.size(), AlwaysTrue{});

  for (std::size_t n = 0; n != array.size(); ++n) {
    FilterIterator const iter(view, Iterator(array.data() + n));
    ValueType& result = *iter;
    ASSERT_SAME_TYPE(ValueType&, decltype(*iter));
    assert(&result == array.data() + n);
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

  test<cpp17_input_iterator<int const*>,   int const>();
  test<cpp20_input_iterator<int const*>,   int const>();
  test<forward_iterator<int const*>,       int const>();
  test<bidirectional_iterator<int const*>, int const>();
  test<random_access_iterator<int const*>, int const>();
  test<contiguous_iterator<int const*>,    int const>();
  test<int const*,                         int const>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
