//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// template<container-compatible-range<value_type> R>
//   void insert_range(R&& rg);

#include <algorithm>
#include <deque>
#include <flat_set>
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

using Set = std::flat_set<int, double>;

static_assert(CanInsertRange<Set, std::ranges::subrange<int*>>);
static_assert(CanInsertRange<Set, std::ranges::subrange<short*>>);
static_assert(!CanInsertRange<Set, std::ranges::subrange<std::pair<int, int>*>>);
static_assert(!CanInsertRange<Set, std::ranges::subrange<std::pair<short, short>*>>);

template <class KeyContainer>
constexpr void test_one() {
  using Key = typename KeyContainer::value_type;

  {
    using M                 = std::flat_set<Key, std::less<Key>, KeyContainer>;
    using It                = forward_iterator<const int*>;
    M m                     = {10, 8, 5, 2, 1};
    int ar[]                = {3, 1, 4, 1, 5, 9};
    std::ranges::subrange r = {It(ar), It(ar + 6)};
    static_assert(std::ranges::common_range<decltype(r)>);
    m.insert_range(r);
    assert((m == M{1, 2, 3, 4, 5, 8, 9, 10}));
  }
  {
    using M                 = std::flat_set<Key, std::greater<>, KeyContainer>;
    using It                = cpp20_input_iterator<const int*>;
    M m                     = {8, 5, 3, 2};
    int ar[]                = {3, 1, 4, 1, 5, 9};
    std::ranges::subrange r = {It(ar), sentinel_wrapper<It>(It(ar + 6))};
    static_assert(!std::ranges::common_range<decltype(r)>);
    m.insert_range(r);
    assert((m == M{1, 2, 3, 4, 5, 8, 9}));
  }
  {
    // The "uniquing" part uses the comparator, not operator==.
    struct ModTen {
      constexpr bool operator()(int a, int b) const { return (a % 10) < (b % 10); }
    };
    using M  = std::flat_set<Key, ModTen, KeyContainer>;
    M m      = {21, 43, 15, 37};
    int ar[] = {33, 18, 55, 18, 42};
    m.insert_range(ar);
    assert((m == M{21, 42, 43, 15, 37, 18}));
  }
  {
    // was empty
    using M = std::flat_set<Key, std::less<Key>, KeyContainer>;
    M m;
    int ar[] = {3, 1, 4, 1, 5, 9};
    m.insert_range(ar);
    assert((m == M{1, 3, 4, 5, 9}));
  }
}

constexpr bool test() {
  test_one<std::vector<int>>();
#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
    test_one<std::deque<int>>();
  test_one<MinSequenceContainer<int>>();
  test_one<std::vector<int, min_allocator<int>>>();
  {
    // Items are forwarded correctly from the input range.
    MoveOnly a[] = {3, 1, 4, 1, 5};
    std::flat_set<MoveOnly> m;
    m.insert_range(a | std::views::as_rvalue);
    MoveOnly expected[] = {1, 3, 4, 5};
    assert(std::ranges::equal(m, expected));
  }

  return true;
}

void test_exception() {
  auto insert_func = [](auto& m, const auto& newValues) { m.insert_range(newValues); };
  test_insert_range_exception_guarantee(insert_func);
}

int main(int, char**) {
  test();
  test_exception();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif

  return 0;
}
