//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// friend bool operator==(const flat_set& x, const flat_set& y);
// friend synth-three-way-result<value_type>
//   operator<=>(const flat_set& x, const flat_set& y);

#include <algorithm>
#include <cassert>
#include <deque>
#include <compare>
#include <flat_set>
#include <functional>
#include <limits>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_comparisons.h"
#include "test_container_comparisons.h"

template <class KeyContainer>
void test() {
  using Key = typename KeyContainer::value_type;

  {
    using C = std::flat_set<Key>;
    C s1    = {1};
    C s2    = {2};
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::strong_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisons(s1, s2, false, true));
    s2 = {1};
    assert(testComparisons(s1, s2, true, false));
    s2 = {1, 2};
    assert(testComparisons(s1, s2, false, true));
    s1 = {0, 1, 2};
    assert(testComparisons(s1, s2, false, true));
    s2 = {0, 1, 3};
    assert(testComparisons(s1, s2, false, true));
  }
  {
    // Comparisons use value_type's native operators, not the comparator
    using C = std::flat_set<Key, std::greater<Key>>;
    C s1    = {1};
    C s2    = {2};
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::strong_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisons(s1, s2, false, true));
    s2 = {1};
    assert(testComparisons(s1, s2, true, false));
    s2 = {1, 2};
    assert(testComparisons(s1, s2, false, true));
    s1 = {0, 1, 2};
    assert(testComparisons(s1, s2, false, false));
    s2 = {0, 1, 3};
    assert(testComparisons(s1, s2, false, true));
  }
}

int main(int, char**) {
  test<std::vector<int>>();
  test<std::deque<int>>();
  test<MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>>();

  {
    using C = std::flat_set<double>;
    C s1    = {1};
    C s2    = C(std::sorted_unique, {std::numeric_limits<double>::quiet_NaN()});
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::partial_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisonsComplete(s1, s2, false, false, false));
  }
  {
    // Comparisons use value_type's native operators, not the comparator
    struct StrongComp {
      bool operator()(double a, double b) const { return std::strong_order(a, b) < 0; }
    };
    using C = std::flat_set<double, StrongComp>;
    C s1    = {1};
    C s2    = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::partial_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisonsComplete(s1, s2, false, false, false));
    s1 = {1, std::numeric_limits<double>::quiet_NaN(), 1};
    s2 = {std::numeric_limits<double>::quiet_NaN(), 1};
    assert(std::lexicographical_compare_three_way(s1.begin(), s1.end(), s2.begin(), s2.end(), std::strong_order) ==
           std::strong_ordering::equal);
    assert(s1 != s2);
    assert((s1 <=> s2) == std::partial_ordering::unordered);
  }
  return 0;
}
