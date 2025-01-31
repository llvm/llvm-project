//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <flat_set>

// flat_set& operator=(flat_set&&);
// Preserves the class invariant for the moved-from flat_set.

#include <algorithm>
#include <cassert>
#include <compare>
#include <flat_set>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"

struct MoveNegates {
  int value_    = 0;
  MoveNegates() = default;
  MoveNegates(int v) : value_(v) {}
  MoveNegates(MoveNegates&& rhs) : value_(rhs.value_) { rhs.value_ = -rhs.value_; }
  MoveNegates& operator=(MoveNegates&& rhs) {
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
  MoveClears(int v) : value_(v) {}
  MoveClears(MoveClears&& rhs) : value_(rhs.value_) { rhs.value_ = 0; }
  MoveClears& operator=(MoveClears&& rhs) {
    value_     = rhs.value_;
    rhs.value_ = 0;
    return *this;
  }
  ~MoveClears()                             = default;
  auto operator<=>(const MoveClears&) const = default;
};

int main(int, char**) {
  {
    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    using M              = std::flat_set<MoveNegates, std::less<MoveNegates>>;
    M m                  = M(expected, expected + 8);
    M m2                 = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.key_comp()));                // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), m.key_comp()) == m.end()); // still contains no duplicates
    m.insert(1);
    m.insert(2);
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    using M              = std::flat_set<MoveClears, std::less<MoveClears>>;
    M m                  = M(expected, expected + 8);
    M m2                 = M(expected, expected + 3);

    m2 = std::move(m);

    assert(std::equal(m2.begin(), m2.end(), expected, expected + 8));
    LIBCPP_ASSERT(m.empty());
    assert(std::is_sorted(m.begin(), m.end(), m.key_comp()));                // still sorted
    assert(std::adjacent_find(m.begin(), m.end(), m.key_comp()) == m.end()); // still contains no duplicates
    m.insert(1);
    m.insert(2);
    assert(m.contains(1));
    assert(m.find(2) != m.end());
  }
  {
    // moved-from object maintains invariant if one of underlying container does not clear after move
    using M = std::flat_set<int, std::less<>, std::vector<int>>;
    M m1    = M({1, 2, 3});
    M m2    = M({1, 2});
    m2      = std::move(m1);
    assert(m2.size() == 3);
    check_invariant(m1);
    LIBCPP_ASSERT(m1.empty());
  }
  return 0;
}
