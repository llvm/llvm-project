//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

#include <algorithm>
#include <cassert>
#include <climits>
#include <deque>
#include <initializer_list>
#include <list>
#include <flat_map>
#include <functional>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "deduction_guides_sfinae_checks.h"
#include "test_allocator.h"

using P  = std::pair<int, long>;
using PC = std::pair<const int, long>;

void test_copy() {
  {
    std::flat_multimap<long, short> source = {{1, 2}, {1, 3}};
    std::flat_multimap s(source);
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    std::flat_multimap<long, short, std::greater<long>> source = {{1, 2}, {1, 3}};
    std::flat_multimap s{source}; // braces instead of parens
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
  {
    std::flat_multimap<long, short, std::greater<long>> source = {{1, 2}, {1, 3}};
    std::flat_multimap s(source, std::allocator<int>());
    ASSERT_SAME_TYPE(decltype(s), decltype(source));
    assert(s == source);
  }
}

void test_containers() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, 2, 2, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> vs({1, 2, 3, 4, 5, 3, 4}, test_allocator<int>(0, 43));
  std::deque<int, test_allocator<int>> sorted_ks({1, 1, 2, 2, 2, 3, INT_MAX}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> sorted_vs({1, 3, 2, 4, 5, 4, 3}, test_allocator<int>(0, 43));
  const std::pair<int, short> expected[] = {{1, 1}, {1, 3}, {2, 2}, {2, 4}, {2, 5}, {3, 4}, {INT_MAX, 3}};
  {
    std::flat_multimap s(ks, vs);

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::less<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 43);
  }
  {
    std::flat_multimap s(std::sorted_equivalent, sorted_ks, sorted_vs);

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::less<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 43);
  }
  {
    std::flat_multimap s(ks, vs, test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::less<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 44);
    assert(s.values().get_allocator().get_id() == 44);
  }
  {
    std::flat_multimap s(std::sorted_equivalent, sorted_ks, sorted_vs, test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::less<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 44);
    assert(s.values().get_allocator().get_id() == 44);
  }
}

void test_containers_compare() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, 2, 2, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> vs({1, 2, 3, 4, 5, 3, 4}, test_allocator<int>(0, 43));
  std::deque<int, test_allocator<int>> sorted_ks({INT_MAX, 3, 2, 2, 2, 1, 1}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> sorted_vs({3, 4, 2, 4, 5, 1, 3}, test_allocator<int>(0, 43));
  const std::pair<int, short> expected[] = {{INT_MAX, 3}, {3, 4}, {2, 2}, {2, 4}, {2, 5}, {1, 1}, {1, 3}};
  {
    std::flat_multimap s(ks, vs, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::greater<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 43);
  }
  {
    std::flat_multimap s(std::sorted_equivalent, sorted_ks, sorted_vs, std::greater<int>());

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::greater<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 43);
  }
  {
    std::flat_multimap s(ks, vs, std::greater<int>(), test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::greater<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 44);
    assert(s.values().get_allocator().get_id() == 44);
  }
  {
    std::flat_multimap s(
        std::sorted_equivalent, sorted_ks, sorted_vs, std::greater<int>(), test_allocator<long>(0, 44));

    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::greater<int>, decltype(ks), decltype(vs)>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 44);
    assert(s.values().get_allocator().get_id() == 44);
  }
}

void test_iter_iter() {
  const P arr[]          = {{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}};
  const P sorted_arr[]   = {{1, 1L}, {1, 1L}, {2, 2L}, {3, 1L}, {INT_MAX, 1L}};
  const PC arrc[]        = {{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}};
  const PC sorted_arrc[] = {{1, 1L}, {1, 1L}, {2, 2L}, {3, 1L}, {INT_MAX, 1L}};
  {
    std::flat_multimap m(std::begin(arr), std::end(arr));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::begin(arrc), std::end(arrc));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc));

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap<int, short> mo;
    std::flat_multimap m(mo.begin(), mo.end());
    ASSERT_SAME_TYPE(decltype(m), decltype(mo));
  }
  {
    std::flat_multimap<int, short> mo;
    std::flat_multimap m(mo.cbegin(), mo.cend());
    ASSERT_SAME_TYPE(decltype(m), decltype(mo));
  }
  {
    std::pair<int, int> source[3] = {{1, 1}, {1, 1}, {3, 3}};
    std::flat_multimap s          = {source, source + 3}; // flat_multimap(InputIterator, InputIterator)
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, int>);
    assert(s.size() == 3);
  }
  {
    std::pair<int, int> source[3] = {{1, 1}, {1, 1}, {3, 3}};
    std::flat_multimap s{source, source + 3}; // flat_multimap(InputIterator, InputIterator)
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, int>);
    assert(s.size() == 3);
  }
  {
    std::pair<int, int> source[3] = {{1, 1}, {1, 2}, {3, 3}};
    std::flat_multimap s{
        std::sorted_equivalent, source, source + 3}; // flat_multimap(sorted_equivalent_t, InputIterator, InputIterator)
    static_assert(std::is_same_v<decltype(s), std::flat_multimap<int, int>>);
    assert(s.size() == 3);
  }
}

void test_iter_iter_compare() {
  const P arr[]          = {{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}};
  const P sorted_arr[]   = {{INT_MAX, 1L}, {3, 1L}, {2, 2L}, {1, 1L}, {1, 1L}};
  const PC arrc[]        = {{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}};
  const PC sorted_arrc[] = {{INT_MAX, 1L}, {3, 1L}, {2, 2L}, {1, 1L}, {1, 1L}};
  using C                = std::greater<long long>;
  {
    std::flat_multimap m(std::begin(arr), std::end(arr), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::begin(arrc), std::end(arrc), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, std::begin(sorted_arr), std::end(sorted_arr), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, std::begin(sorted_arrc), std::end(sorted_arrc), C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap<int, short> mo;
    std::flat_multimap m(mo.begin(), mo.end(), C());
    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, short, C>);
  }
  {
    std::flat_multimap<int, short> mo;
    std::flat_multimap m(mo.cbegin(), mo.cend(), C());
    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, short, C>);
  }
}

void test_initializer_list() {
  const P sorted_arr[] = {{1, 1L}, {1, 1L}, {2, 2L}, {3, 1L}, {INT_MAX, 1L}};
  {
    std::flat_multimap m{std::pair{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}};

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, {std::pair{1, 1L}, {1, 1L}, {2, 2L}, {3, 1L}, {INT_MAX, 1L}});

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap s = {std::make_pair(1, 'a')}; // flat_multimap(initializer_list<pair<int, char>>)
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, char>);
    assert(s.size() == 1);
  }
  {
    using M = std::flat_multimap<int, short>;
    M m;
    std::flat_multimap s = {std::make_pair(m, m)}; // flat_multimap(initializer_list<pair<M, M>>)
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<M, M>);
    assert(s.size() == 1);
    assert(s.find(m)->second == m);
  }
}

void test_initializer_list_compare() {
  const P sorted_arr[] = {{INT_MAX, 1L}, {3, 1L}, {2, 2L}, {1, 1L}, {1, 1L}};
  using C              = std::greater<long long>;
  {
    std::flat_multimap m({std::pair{1, 1L}, {2, 2L}, {1, 1L}, {INT_MAX, 1L}, {3, 1L}}, C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
  {
    std::flat_multimap m(std::sorted_equivalent, {std::pair{INT_MAX, 1L}, {3, 1L}, {2, 2L}, {1, 1L}, {1, 1L}}, C());

    ASSERT_SAME_TYPE(decltype(m), std::flat_multimap<int, long, C>);
    assert(std::ranges::equal(m, sorted_arr));
  }
}

void test_from_range() {
  std::list<std::pair<int, short>> r     = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 4}, {3, 5}};
  const std::pair<int, short> expected[] = {{1, 1}, {1, 1}, {2, 2}, {3, 5}, {INT_MAX, 4}};
  {
    std::flat_multimap s(std::from_range, r);
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::less<int>>);
    assert(std::ranges::equal(s, expected));
  }
  {
    std::flat_multimap s(std::from_range, r, test_allocator<long>(0, 42));
    ASSERT_SAME_TYPE(
        decltype(s),
        std::flat_multimap<int,
                           short,
                           std::less<int>,
                           std::vector<int, test_allocator<int>>,
                           std::vector<short, test_allocator<short>>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 42);
  }
}

void test_from_range_compare() {
  std::list<std::pair<int, short>> r     = {{1, 1}, {2, 2}, {1, 1}, {INT_MAX, 4}, {3, 5}};
  const std::pair<int, short> expected[] = {{INT_MAX, 4}, {3, 5}, {2, 2}, {1, 1}, {1, 1}};
  {
    std::flat_multimap s(std::from_range, r, std::greater<int>());
    ASSERT_SAME_TYPE(decltype(s), std::flat_multimap<int, short, std::greater<int>>);
    assert(std::ranges::equal(s, expected));
  }
  {
    std::flat_multimap s(std::from_range, r, std::greater<int>(), test_allocator<long>(0, 42));
    ASSERT_SAME_TYPE(
        decltype(s),
        std::flat_multimap<int,
                           short,
                           std::greater<int>,
                           std::vector<int, test_allocator<int>>,
                           std::vector<short, test_allocator<short>>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().get_id() == 42);
    assert(s.values().get_allocator().get_id() == 42);
  }
}

int main(int, char**) {
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

  AssociativeContainerDeductionGuidesSfinaeAway<std::flat_multimap, std::flat_multimap<int, short>>();

  return 0;
}
