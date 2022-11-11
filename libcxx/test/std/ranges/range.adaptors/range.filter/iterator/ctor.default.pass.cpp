//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// std::ranges::filter_view<V>::<iterator>() requires default_initializable<iterator_t<V>> = default;

#include <ranges>

#include <cassert>
#include <type_traits>
#include "test_iterators.h"
#include "../types.h"

template <class Iterator, bool IsNoexcept>
constexpr void test_default_constructible() {
  // Make sure the iterator is default constructible when the underlying iterator is.
  using View = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;
  FilterIterator iter1{};
  FilterIterator iter2;
  assert(iter1 == iter2);
  static_assert(noexcept(FilterIterator()) == IsNoexcept);
}

template <class Iterator>
constexpr void test_not_default_constructible() {
  // Make sure the iterator is *not* default constructible when the underlying iterator isn't.
  using View = minimal_view<Iterator, sentinel_wrapper<Iterator>>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterIterator = std::ranges::iterator_t<FilterView>;
  static_assert(!std::is_default_constructible_v<FilterIterator>);
}

constexpr bool tests() {
  test_not_default_constructible<cpp17_input_iterator<int*>>();
  test_not_default_constructible<cpp20_input_iterator<int*>>();
  test_default_constructible<forward_iterator<int*>,         /* noexcept */ false>();
  test_default_constructible<bidirectional_iterator<int*>,   /* noexcept */ false>();
  test_default_constructible<random_access_iterator<int*>,   /* noexcept */ false>();
  test_default_constructible<contiguous_iterator<int*>,      /* noexcept */ false>();
  test_default_constructible<int*,                           /* noexcept */ true>();
  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
