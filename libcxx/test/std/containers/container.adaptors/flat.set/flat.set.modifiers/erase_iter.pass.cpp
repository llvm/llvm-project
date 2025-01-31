//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// iterator erase(iterator position);
// iterator erase(const_iterator position);

#include <compare>
#include <concepts>
#include <deque>
#include <flat_set>
#include <functional>
#include <utility>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "test_macros.h"
#include "min_allocator.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;
  using M   = std::flat_set<Key, std::less<Key>, KeyContainer>;
  using I   = M::iterator;

  int ar[] = {
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
  };
  M m(ar, ar + sizeof(ar) / sizeof(ar[0]));
  assert(m.size() == 8);
  std::same_as<I> decltype(auto) i1 = m.erase(std::next(m.cbegin(), 3));
  assert(m.size() == 7);
  assert(i1 == std::next(m.begin(), 3));
  assert(*m.begin() == 1);
  assert(*std::next(m.begin()) == 2);
  assert(*std::next(m.begin(), 2) == 3);
  assert(*std::next(m.begin(), 3) == 5);
  assert(*std::next(m.begin(), 4) == 6);
  assert(*std::next(m.begin(), 5) == 7);
  assert(*std::next(m.begin(), 6) == 8);

  std::same_as<I> decltype(auto) i2 = m.erase(std::next(m.begin(), 0));
  assert(m.size() == 6);
  assert(i2 == m.begin());
  assert(*m.begin() == 2);
  assert(*std::next(m.begin()) == 3);
  assert(*std::next(m.begin(), 2) == 5);
  assert(*std::next(m.begin(), 3) == 6);
  assert(*std::next(m.begin(), 4) == 7);
  assert(*std::next(m.begin(), 5) == 8);

  std::same_as<I> decltype(auto) i3 = m.erase(std::next(m.cbegin(), 5));
  assert(m.size() == 5);
  assert(i3 == m.end());
  assert(*m.begin() == 2);
  assert(*std::next(m.begin()) == 3);
  assert(*std::next(m.begin(), 2) == 5);
  assert(*std::next(m.begin(), 3) == 6);
  assert(*std::next(m.begin(), 4) == 7);

  std::same_as<I> decltype(auto) i4 = m.erase(std::next(m.begin(), 1));
  assert(m.size() == 4);
  assert(i4 == std::next(m.begin()));
  assert(*m.begin() == 2);
  assert(*std::next(m.begin()) == 5);
  assert(*std::next(m.begin(), 2) == 6);
  assert(*std::next(m.begin(), 3) == 7);

  std::same_as<I> decltype(auto) i5 = m.erase(std::next(m.cbegin(), 2));
  assert(m.size() == 3);
  assert(i5 == std::next(m.begin(), 2));
  assert(*m.begin() == 2);
  assert(*std::next(m.begin()) == 5);
  assert(*std::next(m.begin(), 2) == 7);

  std::same_as<I> decltype(auto) i6 = m.erase(std::next(m.begin(), 2));
  assert(m.size() == 2);
  assert(i6 == std::next(m.begin(), 2));
  assert(*m.begin() == 2);
  assert(*std::next(m.begin()) == 5);

  std::same_as<I> decltype(auto) i7 = m.erase(std::next(m.cbegin(), 0));
  assert(m.size() == 1);
  assert(i7 == std::next(m.begin(), 0));
  assert(*m.begin() == 5);

  std::same_as<I> decltype(auto) i8 = m.erase(m.begin());
  assert(m.size() == 0);
  assert(i8 == m.begin());
  assert(i8 == m.end());
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  {
    auto erase_function = [](auto& m, auto) { m.erase(m.begin() + 2); };
    test_erase_exception_guarantee(erase_function);
  }

  return 0;
}
