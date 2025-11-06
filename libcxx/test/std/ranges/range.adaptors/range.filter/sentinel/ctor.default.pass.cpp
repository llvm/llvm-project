//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// filter_view<V>::<sentinel>() = default;

#include <ranges>

#include <cassert>
#include "test_iterators.h"
#include "../types.h"

template <class Iter, class Sent = sentinel_wrapper<Iter>>
constexpr void test() {
  using View = minimal_view<Iter, Sent>;
  using FilterView = std::ranges::filter_view<View, AlwaysTrue>;
  using FilterSentinel = std::ranges::sentinel_t<FilterView>;
  FilterSentinel sent1{};
  FilterSentinel sent2;
  assert(base(base(sent1.base())) == base(base(sent2.base())));
  static_assert(noexcept(FilterSentinel()));
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
