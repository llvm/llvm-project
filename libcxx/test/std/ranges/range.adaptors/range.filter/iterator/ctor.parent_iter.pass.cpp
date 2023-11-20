//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// constexpr std::ranges::filter_view::<iterator>(filter_view&, iterator_t<V>);

#include <ranges>

#include <array>
#include <cassert>
#include <utility>
#include "test_iterators.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View = minimal_view<Iterator, Sentinel>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;

  std::array<int, 10> array{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  View view(Iterator(array.data()), Sentinel(Iterator(array.data() + array.size())));
  Iterator iter = view.begin();

  FilterView filter_view(std::move(view), AlwaysTrue{});
  FilterIterator filter_iter(filter_view, std::move(iter));
  assert(base(filter_iter.base()) == array.data());
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
