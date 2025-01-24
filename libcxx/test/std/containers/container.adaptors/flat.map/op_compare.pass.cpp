//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// friend bool operator==(const flat_map& x, const flat_map& y);
// friend synth-three-way-result<value_type>
//   operator<=>(const flat_map& x, const flat_map& y);

#include <algorithm>
#include <cassert>
#include <deque>
#include <compare>
#include <flat_map>
#include <functional>
#include <limits>
#include <vector>

#include "MinSequenceContainer.h"
#include "test_macros.h"
#include "min_allocator.h"
#include "test_allocator.h"
#include "test_comparisons.h"
#include "test_container_comparisons.h"

template <class KeyContainer, class ValueContainer>
void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;

  {
    using C = std::flat_map<Key, Value>;
    C s1    = {{1, 1}};
    C s2    = {{2, 0}}; // {{1,1}} versus {{2,0}}
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::strong_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisons(s1, s2, false, true));
    s2 = {{1, 1}}; // {{1,1}} versus {{1,1}}
    assert(testComparisons(s1, s2, true, false));
    s2 = {{1, 1}, {2, 0}}; // {{1,1}} versus {{1,1},{2,0}}
    assert(testComparisons(s1, s2, false, true));
    s1 = {{0, 0}, {1, 1}, {2, 2}}; // {{0,0},{1,1},{2,2}} versus {{1,1},{2,0}}
    assert(testComparisons(s1, s2, false, true));
    s2 = {{0, 0}, {1, 1}, {2, 3}}; // {{0,0},{1,1},{2,2}} versus {{0,0},{1,1},{2,3}}
    assert(testComparisons(s1, s2, false, true));
  }
  {
    // Comparisons use value_type's native operators, not the comparator
    using C = std::flat_map<Key, Value, std::greater<Key>>;
    C s1    = {{1, 1}};
    C s2    = {{2, 0}}; // {{1,1}} versus {{2,0}}
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::strong_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisons(s1, s2, false, true));
    s2 = {{1, 1}}; // {{1,1}} versus {{1,1}}
    assert(testComparisons(s1, s2, true, false));
    s2 = {{1, 1}, {2, 0}}; // {{1,1}} versus {{2,0},{1,1}}
    assert(testComparisons(s1, s2, false, true));
    s1 = {{0, 0}, {1, 1}, {2, 2}}; // {{2,2},{1,1},{0,0}} versus {2,0},{1,1}}
    assert(testComparisons(s1, s2, false, false));
    s2 = {{0, 0}, {1, 1}, {2, 3}}; // {{2,2},{1,1},{0,0}} versus {{2,3},{1,1},{0,0}}
    assert(testComparisons(s1, s2, false, true));
  }
}

int main(int, char**) {
  test<std::vector<int>, std::vector<int>>();
  test<std::deque<int>, std::deque<int>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();

  {
    using C = std::flat_map<double, int>;
    C s1    = {{1, 1}};
    C s2    = C(std::sorted_unique, {{std::numeric_limits<double>::quiet_NaN(), 2}});
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::partial_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisonsComplete(s1, s2, false, false, false));
  }
  {
    using C = std::flat_map<int, double>;
    C s1    = {{1, 1}};
    C s2    = C(std::sorted_unique, {{2, std::numeric_limits<double>::quiet_NaN()}});
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::partial_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisonsComplete(s1, s2, false, true, false));
    s2 = C(std::sorted_unique, {{1, std::numeric_limits<double>::quiet_NaN()}});
    assert(testComparisonsComplete(s1, s2, false, false, false));
  }
  {
    // Comparisons use value_type's native operators, not the comparator
    struct StrongComp {
      bool operator()(double a, double b) const { return std::strong_order(a, b) < 0; }
    };
    using C = std::flat_map<double, double, StrongComp>;
    C s1    = {{1, 1}};
    C s2    = {{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()}};
    ASSERT_SAME_TYPE(decltype(s1 <=> s2), std::partial_ordering);
    AssertComparisonsReturnBool<C>();
    assert(testComparisonsComplete(s1, s2, false, false, false));
    s1 = {{{1, 1}, {std::numeric_limits<double>::quiet_NaN(), 1}}};
    s2 = {{{std::numeric_limits<double>::quiet_NaN(), 1}, {1, 1}}};
    assert(std::lexicographical_compare_three_way(
               s1.keys().begin(), s1.keys().end(), s2.keys().begin(), s2.keys().end(), std::strong_order) ==
           std::strong_ordering::equal);
    assert(s1 != s2);
    assert((s1 <=> s2) == std::partial_ordering::unordered);
  }
  return 0;
}
