//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <ranges>
#include <string>
#include <vector>
#include "MinSequenceContainer.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  {
    using Key = typename KeyContainer::value_type;
    using C   = std::flat_set<Key, std::less<Key>, KeyContainer>;

    static_assert(std::same_as<std::ranges::iterator_t<C>, typename C::iterator>);
    static_assert(std::ranges::random_access_range<C>);
    static_assert(std::ranges::common_range<C>);
    static_assert(std::ranges::input_range<C>);
    static_assert(!std::ranges::view<C>);
    static_assert(std::ranges::sized_range<C>);
    static_assert(!std::ranges::borrowed_range<C>);
    static_assert(std::ranges::viewable_range<C>);

    static_assert(std::same_as<std::ranges::iterator_t<const C>, typename C::const_iterator>);
    static_assert(std::ranges::random_access_range<const C>);
    static_assert(std::ranges::common_range<const C>);
    static_assert(std::ranges::input_range<const C>);
    static_assert(!std::ranges::view<const C>);
    static_assert(std::ranges::sized_range<const C>);
    static_assert(!std::ranges::borrowed_range<const C>);
    static_assert(!std::ranges::viewable_range<const C>);
  }
}

void test() {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();
}
