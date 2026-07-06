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

// template<container-compatible-range<value_type> R>
//   void insert_range(R&& rg);

#include <algorithm>
#include <deque>
#include <flat_map>
#include <functional>
#include <ranges>
#include <vector>

#include "MinSequenceContainer.h"
#include "../helpers.h"
#include "MoveOnly.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

// test constraint container-compatible-range
template <class M, class R>
concept CanInsertRange = requires(M m, R&& r) { m.insert_range(std::forward<R>(r)); };

using Map = std::flat_multimap<int, double>;

static_assert(CanInsertRange<Map, std::ranges::subrange<std::pair<int, double>*>>);
static_assert(CanInsertRange<Map, std::ranges::subrange<std::pair<short, double>*>>);
static_assert(!CanInsertRange<Map, std::ranges::subrange<int*>>);
static_assert(!CanInsertRange<Map, std::ranges::subrange<double*>>);

template <class KeyContainer, class ValueContainer>
constexpr void test() {
  using Key   = typename KeyContainer::value_type;
  using Value = typename ValueContainer::value_type;

  {
    using P                 = std::pair<int, int>;
    using M                 = std::flat_multimap<Key, Value, std::less<Key>, KeyContainer, ValueContainer>;
    using It                = forward_iterator<const P*>;
    M m                     = {{10, 1}, {8, 2}, {5, 3}, {2, 4}, {1, 5}};
    P ar[]                  = {{3, 1}, {1, 2}, {4, 3}, {1, 4}, {5, 5}, {9, 6}};
    std::ranges::subrange r = {It(ar), It(ar + 6)};
    static_assert(std::ranges::common_range<decltype(r)>);
    m.insert_range(r);
    std::vector<P> expected = {{1, 5}, {1, 2}, {1, 4}, {2, 4}, {3, 1}, {4, 3}, {5, 3}, {5, 5}, {8, 2}, {9, 6}, {10, 1}};
    assert(std::ranges::equal(m, expected));
  }
  {
    using P                 = std::pair<int, int>;
    using M                 = std::flat_multimap<Key, Value, std::greater<>, KeyContainer, ValueContainer>;
    using It                = cpp20_input_iterator<const P*>;
    M m                     = {{8, 1}, {5, 2}, {3, 3}, {2, 4}};
    P ar[]                  = {{3, 1}, {1, 2}, {4, 3}, {1, 4}, {5, 5}, {9, 6}};
    std::ranges::subrange r = {It(ar), sentinel_wrapper<It>(It(ar + 6))};
    static_assert(!std::ranges::common_range<decltype(r)>);
    m.insert_range(r);
    std::vector<P> expected = {{9, 6}, {8, 1}, {5, 2}, {5, 5}, {4, 3}, {3, 3}, {3, 1}, {2, 4}, {1, 2}, {1, 4}};
    assert(std::ranges::equal(m, expected));
  }
}

constexpr bool test() {
  test<std::vector<int>, std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test<std::deque<int>, std::vector<int>>();
  test<MinSequenceContainer<int>, MinSequenceContainer<int>>();
  test<std::vector<int, min_allocator<int>>, std::vector<int, min_allocator<int>>>();
  {
    // Items are forwarded correctly from the input range (P2767).
    std::pair<MoveOnly, MoveOnly> a[] = {{3, 3}, {1, 1}, {4, 4}, {1, 1}, {5, 5}};
    std::flat_multimap<MoveOnly, MoveOnly> m;
    m.insert_range(a | std::views::as_rvalue);
    std::pair<MoveOnly, MoveOnly> expected[] = {{1, 1}, {1, 1}, {3, 3}, {4, 4}, {5, 5}};
    assert(std::ranges::equal(m, expected));
  }
  {
    // The element type of the range doesn't need to be std::pair (P2767).
    std::pair<int, int> pa[] = {{3, 3}, {1, 1}, {4, 4}, {1, 1}, {5, 5}};
    std::vector<std::reference_wrapper<std::pair<int, int>>> a(pa, pa + 5);
    std::flat_multimap<int, int> m;
    m.insert_range(a);
    std::pair<int, int> expected[] = {{1, 1}, {1, 1}, {3, 3}, {4, 4}, {5, 5}};
    assert(std::ranges::equal(m, expected));
  }
  if (!TEST_IS_CONSTANT_EVALUATED) {
    auto insert_func = [](auto& m, const auto& newValues) { m.insert_range(newValues); };
    test_insert_range_exception_guarantee(insert_func);
  }
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
