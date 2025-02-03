//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: availability-pmr-missing

// <flat_map>

#include <algorithm>
#include <cassert>
#include <climits>
#include <deque>
#include <initializer_list>
#include <list>
#include <flat_map>
#include <functional>
#include <memory_resource>
#include <ranges>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_allocator.h"

using P  = std::pair<int, long>;
using PC = std::pair<const int, long>;

void test_containers() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, 2, 2, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> vs({1, 2, 3, 4, 5, 3, 4}, test_allocator<int>(0, 43));
  std::deque<int, test_allocator<int>> sorted_ks({1, 1, 2, 2, 2, 3, INT_MAX}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> sorted_vs({1, 3, 2, 4, 5, 4, 3}, test_allocator<int>(0, 43));
  const std::pair<int, short> expected[] = {{1, 1}, {1, 3}, {2, 2}, {2, 4}, {2, 5}, {3, 4}, {INT_MAX, 3}};
  {
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::monotonic_buffer_resource mr2;
    std::pmr::deque<int> pks(ks.begin(), ks.end(), &mr);
    std::pmr::deque<short> pvs(vs.begin(), vs.end(), &mr);
    std::flat_multimap s(std::move(pks), std::move(pvs), &mr2);

    ASSERT_SAME_TYPE(
        decltype(s), std::flat_multimap<int, short, std::less<int>, std::pmr::deque<int>, std::pmr::deque<short>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().resource() == &mr2);
    assert(s.values().get_allocator().resource() == &mr2);
  }
  {
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::monotonic_buffer_resource mr2;
    std::pmr::deque<int> pks(sorted_ks.begin(), sorted_ks.end(), &mr);
    std::pmr::deque<short> pvs(sorted_vs.begin(), sorted_vs.end(), &mr);
    std::flat_multimap s(std::sorted_equivalent, std::move(pks), std::move(pvs), &mr2);

    ASSERT_SAME_TYPE(
        decltype(s), std::flat_multimap<int, short, std::less<int>, std::pmr::deque<int>, std::pmr::deque<short>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().resource() == &mr2);
    assert(s.values().get_allocator().resource() == &mr2);
  }
}

void test_containers_compare() {
  std::deque<int, test_allocator<int>> ks({1, 2, 1, 2, 2, INT_MAX, 3}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> vs({1, 2, 3, 4, 5, 3, 4}, test_allocator<int>(0, 43));
  std::deque<int, test_allocator<int>> sorted_ks({INT_MAX, 3, 2, 2, 2, 1, 1}, test_allocator<int>(0, 42));
  std::deque<short, test_allocator<short>> sorted_vs({3, 4, 2, 4, 5, 1, 3}, test_allocator<int>(0, 43));
  const std::pair<int, short> expected[] = {{INT_MAX, 3}, {3, 4}, {2, 2}, {2, 4}, {2, 5}, {1, 1}, {1, 3}};

  {
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::monotonic_buffer_resource mr2;
    std::pmr::deque<int> pks(ks.begin(), ks.end(), &mr);
    std::pmr::deque<short> pvs(vs.begin(), vs.end(), &mr);
    std::flat_multimap s(std::move(pks), std::move(pvs), std::greater<int>(), &mr2);

    ASSERT_SAME_TYPE(
        decltype(s), std::flat_multimap<int, short, std::greater<int>, std::pmr::deque<int>, std::pmr::deque<short>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().resource() == &mr2);
    assert(s.values().get_allocator().resource() == &mr2);
  }
  {
    std::pmr::monotonic_buffer_resource mr;
    std::pmr::monotonic_buffer_resource mr2;
    std::pmr::deque<int> pks(sorted_ks.begin(), sorted_ks.end(), &mr);
    std::pmr::deque<short> pvs(sorted_vs.begin(), sorted_vs.end(), &mr);
    std::flat_multimap s(std::sorted_equivalent, std::move(pks), std::move(pvs), std::greater<int>(), &mr2);

    ASSERT_SAME_TYPE(
        decltype(s), std::flat_multimap<int, short, std::greater<int>, std::pmr::deque<int>, std::pmr::deque<short>>);
    assert(std::ranges::equal(s, expected));
    assert(s.keys().get_allocator().resource() == &mr2);
    assert(s.values().get_allocator().resource() == &mr2);
  }
}

int main(int, char**) {
  test_containers();
  test_containers_compare();

  return 0;
}
