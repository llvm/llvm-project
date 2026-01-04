//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_map>

// flat_multimap& operator=(flat_multimap&&);
// Preserves the class invariant for the moved-from flat_multimap.

#include <algorithm>
#include <cassert>
#include <compare>
#include <deque>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"

struct MoveNegates {
  int value_    = 0;
  MoveNegates() = default;
  constexpr MoveNegates(int v) : value_(v) {}
  constexpr MoveNegates(MoveNegates&& rhs) : value_(rhs.value_) { rhs.value_ = -rhs.value_; }
  constexpr MoveNegates& operator=(MoveNegates&& rhs) {
    value_     = rhs.value_;
    rhs.value_ = -rhs.value_;
    return *this;
  }
  ~MoveNegates()                             = default;
  auto operator<=>(const MoveNegates&) const = default;
};

struct MoveClears {
  int value_   = 0;
  MoveClears() = default;
  constexpr MoveClears(int v) : value_(v) {}
  constexpr MoveClears(MoveClears&& rhs) : value_(rhs.value_) { rhs.value_ = 0; }
  constexpr MoveClears& operator=(MoveClears&& rhs) {
    value_     = rhs.value_;
    rhs.value_ = 0;
    return *this;
  }
  ~MoveClears()                             = default;
  auto operator<=>(const MoveClears&) const = default;
};

template <template <class...> class KeyContainer, template <class...> class ValueContainer>
constexpr void test() {
  auto value_eq = [](auto&& p, auto&& q) { return p.first == q.first; };
  {
    const std::pair<int, int> expected[] = {{1, 1}, {1, 2}, {3, 3}, {3, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
    using M =
        std::flat_multimap<MoveNegates, int, std::less<MoveNegates>, KeyContainer<MoveNegates>, ValueContainer<int>>;
    M m  = M(std::sorted_equivalent, expected, expected + 8);
    M m2 = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));          // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), value_eq) == m.end()); // still contains no duplicates
    m.insert({1, 1});
    m.insert({2, 2});
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    const std::pair<int, int> expected[] = {{1, 1}, {1, 2}, {3, 3}, {4, 4}, {4, 5}, {6, 6}, {7, 7}, {8, 8}};
    using M = std::flat_multimap<MoveClears, int, std::less<MoveClears>, KeyContainer<MoveClears>, ValueContainer<int>>;
    M m     = M(std::sorted_equivalent, expected, expected + 8);
    M m2    = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.value_comp()));          // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), value_eq) == m.end()); // still contains no duplicates
    m.insert({1, 1});
    m.insert({2, 2});
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_multimap<int, int, std::less<>, KeyContainer<int>, CopyOnlyVector<int>>;
    M m1    = M({1, 1, 2, 3}, {1, 1, 2, 3});
    M m2    = M({1, 2, 2}, {1, 2, 2});
    m2      = std::move(m1);
    assert(m2.size() == 4);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
    LIBCPP_ASSERT(m1.keys().size() == 0);
    LIBCPP_ASSERT(m1.values().size() == 0);
  }
}

constexpr bool test() {
  test<std::vector, std::vector>();

#ifndef __cpp_lib_constexpr_deque
  if (!TEST_IS_CONSTANT_EVALUATED)
#endif
  {
    test<std::deque, std::deque>();
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
