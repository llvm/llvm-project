//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

#include <algorithm>
#include <cassert>
#include <climits>
#include <deque>
#include <initializer_list>
#include <list>
#include <flat_set>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

void test_copy() {
  {
    std::flat_multiset<long> source = {1, 2, 2};
    std::flat_multiset s(source);
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    std::flat_multiset<short, std::greater<short>> source = {1, 2, 2};
    std::flat_multiset s{source}; // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    std::flat_multiset<long, std::greater<long>> source = {1, 2, 2};
    std::flat_multiset s(source, std::allocator<int>());
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
}
void test_containers() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<int, test_allocator<int>> sorted_ks({1, 1, 2, 3, INT_MAX}, test_allocator<int>(0, 42));
  int expected[] = {1, 1, 2, 3, INT_MAX};
  {
    std::flat_multiset s(ks);

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
  {
    std::flat_multiset s(std::sorted_equivalent, sorted_ks);

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
  {
    std::flat_multiset s(ks, test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 44);
  }
  {
    std::flat_multiset s(std::sorted_equivalent, sorted_ks, test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 44);
  }
}

void test_containers_compare() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<int, test_allocator<int>> sorted_ks({INT_MAX, 3, 2, 1, 1}, test_allocator<int>(0, 42));
  int expected[] = {INT_MAX, 3, 2, 1, 1};
  {
    std::flat_multiset s(ks, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
  {
    std::flat_multiset s(std::sorted_equivalent, sorted_ks, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
  {
    std::flat_multiset s(ks, std::greater<int>(), test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 44);
  }
  {
    std::flat_multiset s(std::sorted_equivalent, sorted_ks, std::greater<int>(), test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, decltype(ks)>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 44);
  }
}

void test_iter_iter() {
  int arr[]               = {1, 2, 1, INT_MAX, 3};
  int sorted_arr[]        = {1, 1, 2, 3, INT_MAX};
  const int arrc[]        = {1, 2, 1, INT_MAX, 3};
  const int sorted_arrc[] = {1, 1, 2, 3, INT_MAX};
  {
    std::flat_multiset m(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::begin(arrc), std::end(arrc));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset<int> mo;
    std::flat_multiset m(mo.begin(), mo.end());
    ASSERT_SAME_TYPE(decltype(m), decltype(mo));
  }
  {
    std::flat_multiset<int> mo;
    std::flat_multiset m(mo.cbegin(), mo.cend());
    ASSERT_SAME_TYPE(decltype(m), decltype(mo));
  }
  {
    int source[3] = {1, 2, 3};
    std::flat_multiset s(source, source + 3);
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int>);
    assert(s.size() == 3);
  }
  {
    // This does not deduce to flat_multiset(InputIterator, InputIterator)
    // But deduces to flat_multiset(initializer_list<int*>)
    int source[3]        = {1, 2, 3};
    std::flat_multiset s = {source, source + 3};
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int*>);
    assert(s.size() == 2);
  }
  {
    int source[3] = {1, 2, 3};
    std::flat_multiset s{
        std::sorted_equivalent, source, source + 3}; // flat_multiset(sorted_equivalent_t, InputIterator, InputIterator)
    static_assert(std::is_same_v<decltype(s), std::flat_multiset<int>>);
    assert(s.size() == 3);
  }
}

void test_iter_iter_compare() {
  int arr[]               = {1, 2, 1, INT_MAX, 3};
  int sorted_arr[]        = {INT_MAX, 3, 2, 1, 1};
  const int arrc[]        = {1, 2, 1, INT_MAX, 3};
  const int sorted_arrc[] = {INT_MAX, 3, 2, 1, 1};
  using C                 = std::greater<long>;
  {
    std::flat_multiset m(std::begin(arr), std::end(arr), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::begin(arrc), std::end(arrc), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset<int> mo;
    std::flat_multiset m(mo.begin(), mo.end(), C());
    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
  }
  {
    std::flat_multiset<int> mo;
    std::flat_multiset m(mo.cbegin(), mo.cend(), C());
    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
  }
}

void test_initializer_list() {
  const int sorted_arr[] = {1, 1, 2, 3, INT_MAX};
  {
    std::flat_multiset m{1, 2, 1, INT_MAX, 3};

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, {1, 1, 2, 3, INT_MAX});

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset s = {1};
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int>);
    assert(s.size() == 1);
  }
  {
    using M = std::flat_multiset<int>;
    M m;
    std::flat_multiset s{m, m}; // flat_multiset(initializer_list<M>)
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<M>);
    assert(s.size() == 2);
  }
}

void test_initializer_list_compare() {
  const int sorted_arr[] = {INT_MAX, 3, 2, 1, 1};
  using C                = std::greater<long>;
  {
    std::flat_multiset m({1, 2, 1, INT_MAX, 3}, C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multiset m(std::sorted_equivalent, {INT_MAX, 3, 2, 1, 1}, C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multiset<int, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
}

void test_from_range() {
  std::list<int> r     = {1, 2, 1, INT_MAX, 3};
  const int expected[] = {1, 1, 2, 3, INT_MAX};
  {
    std::flat_multiset s(std::from_range, r);
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>>);
    assert(std::ranges::equal(s, expected));
  }
  {
    std::flat_multiset s(std::from_range, r, test_allocator<long>(0, 42));
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::less<int>, std::vector<int, test_allocator<int>>>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
}

void test_from_range_compare() {
  std::list<int> r     = {1, 2, 1, INT_MAX, 3};
  const int expected[] = {INT_MAX, 3, 2, 1, 1};
  {
    std::flat_multiset s(std::from_range, r, std::greater<int>());
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>>);
    assert(std::ranges::equal(s, expected));
  }
  {
    std::flat_multiset s(std::from_range, r, std::greater<int>(), test_allocator<long>(0, 42));
    ASSERT_SAME_TYPE(decltype(s), std::flat_multiset<int, std::greater<int>, std::vector<int, test_allocator<int>>>);
    assert(std::ranges::equal(s, expected));
    assert(std::move(s).extract().get_allocator().get_id() == 42);
  }
}

void test() {
  // Each test function also tests the sorted_equivalent-prefixed and allocator-suffixed overloads.
  test_copy();
  test_containers();
  test_containers_compare();
  test_iter_iter();
  test_iter_iter_compare();
  test_initializer_list();
  test_initializer_list_compare();
  test_from_range();
  test_from_range_compare();

  AssociativeContainerDeductionGuidesSfinaeAway<std::flat_multiset, std::flat_multiset<int>>();
}

int main(int, char**) {
  test();

  return 0;
}
