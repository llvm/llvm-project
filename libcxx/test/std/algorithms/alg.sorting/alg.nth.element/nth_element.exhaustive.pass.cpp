//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// Exhaustively check std::nth_element on all permutations of small ranges.
// This specifically covers the size-0..5 sorting-network specializations and the
// size-6..7 selection-sort fallback.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, class Compare>
void check_postcondition(Iter first, Iter nth, Iter last, Compare comp) {
  if (nth == last)
    return;
  for (Iter i = first; i != nth; ++i)
    assert(!comp(*nth, *i));
  for (Iter i = nth; i != last; ++i)
    assert(!comp(*i, *nth));
}

template <class Iter, class Compare>
void test_permutations(std::vector<int> values, Compare comp) {
  std::size_t const size    = values.size();
  std::vector<int> expected = values;
  std::sort(expected.begin(), expected.end(), comp);

  do {
    for (std::size_t nth = 0; nth <= size; ++nth) {
      std::vector<int> v = values;
      std::nth_element(Iter(v.data()), Iter(v.data() + nth), Iter(v.data() + size), comp);
      if (nth != size)
        assert(v[nth] == expected[nth]);
      assert(std::is_permutation(v.begin(), v.end(), values.begin()));
      check_postcondition(Iter(v.data()), Iter(v.data() + nth), Iter(v.data() + size), comp);
    }
  } while (std::next_permutation(values.begin(), values.end()));
}

template <class Iter>
void test_distinct() {
  for (int size = 0; size <= 8; ++size) {
    std::vector<int> values(static_cast<std::size_t>(size));
    std::iota(values.begin(), values.end(), 0);
    test_permutations<Iter>(values, std::less<int>());
    test_permutations<Iter>(values, std::greater<int>());
  }
}

// Exercise equal-element paths with every bit pattern of 0/1 for sizes 1..8.
template <class Iter>
void test_duplicates() {
  for (int size = 1; size <= 8; ++size) {
    for (int mask = 0; mask < (1 << size); ++mask) {
      std::vector<int> values(static_cast<std::size_t>(size));
      for (int i = 0; i < size; ++i)
        values[static_cast<std::size_t>(i)] = (mask >> i) & 1;

      std::vector<int> expected = values;
      std::sort(expected.begin(), expected.end());
      for (int nth = 0; nth < size; ++nth) {
        std::vector<int> v = values;
        std::nth_element(Iter(v.data()), Iter(v.data() + nth), Iter(v.data() + size));
        assert(v[static_cast<std::size_t>(nth)] == expected[static_cast<std::size_t>(nth)]);
        check_postcondition(Iter(v.data()), Iter(v.data() + nth), Iter(v.data() + size), std::less<int>());
      }
    }
  }
}

#if TEST_STD_VER >= 20
constexpr bool test_constexpr_small_ranges() {
  // Sizes 3, 4 and 5 hit the specialized sorting-network dispatch.
  {
    int a[] = {2, 0, 1};
    std::nth_element(a, a + 1, a + 3);
    if (a[1] != 1)
      return false;
  }
  {
    int a[] = {3, 0, 2, 1};
    std::nth_element(a, a + 2, a + 4);
    if (a[2] != 2)
      return false;
  }
  {
    int a[] = {4, 2, 0, 3, 1};
    std::nth_element(a, a + 2, a + 5);
    if (a[2] != 2)
      return false;
  }
  {
    int a[] = {4, 2, 0, 3, 1};
    std::nth_element(a, a + 2, a + 5, std::greater<>());
    if (a[2] != 2)
      return false;
  }
  return true;
}
static_assert(test_constexpr_small_ranges());
#endif

int main(int, char**) {
  test_distinct<int*>();
  test_distinct<random_access_iterator<int*> >();
  test_duplicates<int*>();
  test_duplicates<random_access_iterator<int*> >();

  return 0;
}
