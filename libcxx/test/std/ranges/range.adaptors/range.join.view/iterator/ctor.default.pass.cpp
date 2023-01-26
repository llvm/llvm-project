//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// iterator() requires default_initializable<OuterIter> = default;

#include <ranges>

#include <cassert>
#include <type_traits>
#include "test_iterators.h"
#include "../types.h"

template <class It>
struct view : std::ranges::view_base {
  It begin() const;
  sentinel_wrapper<It> end() const;
};

template <class It>
constexpr void test_default_constructible() {
  using JoinView = std::ranges::join_view<view<It>>;
  using JoinIterator = std::ranges::iterator_t<JoinView>;
  static_assert(std::is_default_constructible_v<JoinIterator>);
  JoinIterator it; (void)it;
}

template <class It>
constexpr void test_non_default_constructible() {
  using JoinView = std::ranges::join_view<view<It>>;
  using JoinIterator = std::ranges::iterator_t<JoinView>;
  static_assert(!std::is_default_constructible_v<JoinIterator>);
}

constexpr bool test() {
  test_non_default_constructible<cpp17_input_iterator<ChildView*>>();
  // NOTE: cpp20_input_iterator can't be used with join_view because it is not copyable.
  test_default_constructible<forward_iterator<ChildView*>>();
  test_default_constructible<bidirectional_iterator<ChildView*>>();
  test_default_constructible<random_access_iterator<ChildView*>>();
  test_default_constructible<contiguous_iterator<ChildView*>>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
