//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// class flat_multimap

// iterator erase(const_iterator first, const_iterator last);

#include <compare>
#include <concepts>
#include <deque>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;
  using M     = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
  using P     = std::pair<Key, Value>;
  using I     = M::iterator;

  P ar[] = {
      P(1, 1.5),
      P(2, 2.5),
      P(2, 2.6),
      P(3, 3.5),
      P(3, 3.6),
      P(3, 3.7),
      P(4, 4.5),
      P(5, 5.5),
      P(6, 6.5),
      P(7, 7.5),
      P(8, 8.5),
  };
  M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
  assert(m.size() == 11);
  std::same_as<I> decltype(auto) i1 = m.erase(m.cbegin(), m.cbegin());
  assert(m.size() == 11);
  assert(i1 == m.begin());
  assert(std::ranges::equal(
      m,
      std::vector<P>{
          {1, 1.5},
          {2, 2.5},
          {2, 2.6},
          {3, 3.5},
          {3, 3.6},
          {3, 3.7},
          {4, 4.5},
          {5, 5.5},
          {6, 6.5},
          {7, 7.5},
          {8, 8.5}}));

  std::same_as<I> decltype(auto) i2 = m.erase(m.cbegin(), std::next(m.cbegin(), 2));
  assert(m.size() == 9);
  assert(i2 == m.begin());
  assert(std::ranges::equal(
      m, std::vector<P>{{2, 2.6}, {3, 3.5}, {3, 3.6}, {3, 3.7}, {4, 4.5}, {5, 5.5}, {6, 6.5}, {7, 7.5}, {8, 8.5}}));

  std::same_as<I> decltype(auto) i3 = m.erase(std::next(m.cbegin(), 2), std::next(m.cbegin(), 6));
  assert(m.size() == 5);
  assert(i3 == std::next(m.begin(), 2));
  assert(std::ranges::equal(m, std::vector<P>{{2, 2.6}, {3, 3.5}, {6, 6.5}, {7, 7.5}, {8, 8.5}}));

  std::same_as<I> decltype(auto) i4 = m.erase(m.cbegin(), m.cend());
  assert(m.size() == 0);
  assert(i4 == m.begin());
  assert(i4 == m.end());
}

int main(int, char**) {
  test<std::vector<int>, std::vector<double>>();
  test<std::deque<int>, std::vector<double>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<double>>();
  test<std::vector<int, min_allocator<int>>, std::vector<double, min_allocator<double>>>();

  {
    auto erase_function = [](auto& m, auto) { m.erase(m.begin(), m.begin() + 2); };
    test_erase_exception_guarantee(erase_function);
  }
  return 0;
}
