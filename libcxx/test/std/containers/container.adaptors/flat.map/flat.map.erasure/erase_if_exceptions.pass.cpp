//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-exceptions

// <flat_map>

// template<class Key, class T, class Compare, class KeyContainer, class MappedContainer, class Predicate>
//   typename flat_map<Key, T, Compare, KeyContainer, MappedContainer>::size_type
//   erase_if(flat_map<Key, T, Compare, KeyContainer, MappedContainer>& c, Predicate pred);
// If any member function in [flat.set.defn] exits via an exception, the invariant is restored.
// (This is not a member function, but let's respect the invariant anyway.)

#include <algorithm>
#include <cassert>
#include <deque>
#include <flat_map>
#include <functional>
#include <utility>
#include <vector>

#include "../helpers.h"
#include "test_macros.h"

struct Counter {
  int c1, c2, throws;
  void tick() {
    c1 -= 1;
    if (c1 == 0) {
      c1 = c2;
      throws += 1;
      throw 42;
    }
  }
};
Counter g_counter = {0, 0, 0};

struct ThrowingAssignment {
  ThrowingAssignment(int i) : i_(i) {}
  ThrowingAssignment(const ThrowingAssignment&) = default;
  ThrowingAssignment& operator=(const ThrowingAssignment& rhs) {
    g_counter.tick();
    i_ = rhs.i_;
    g_counter.tick();
    return *this;
  }
  operator int() const { return i_; }
  int i_;
};

struct ThrowingComparator {
  bool operator()(const ThrowingAssignment& a, const ThrowingAssignment& b) const {
    g_counter.tick();
    return a.i_ < b.i_;
  }
};

struct ErasurePredicate {
  bool operator()(const auto& x) const { return (3 <= x.first && x.first <= 5); }
};

int main(int, char**) {
  const std::pair<int, int> expected[] = {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}, {8, 8}};
  {
    using M = std::flat_map<ThrowingAssignment, int, ThrowingComparator>;
    for (int first_throw = 1; first_throw < 99; ++first_throw) {
      for (int second_throw = 1; second_throw < 99; ++second_throw) {
        g_counter = {0, 0, 0};
        M m       = M({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
        try {
          g_counter = {first_throw, second_throw, 0};
          auto n    = std::erase_if(m, ErasurePredicate());
          assert(n == 3);
          // If it didn't throw at all, we're done.
          g_counter = {0, 0, 0};
          assert((m == M{{1, 1}, {2, 2}, {6, 6}, {7, 7}, {8, 8}}));
          first_throw = 99; // "done"
          break;
        } catch (int ex) {
          assert(ex == 42);
          check_invariant(m);
          LIBCPP_ASSERT(m.empty() || std::equal(m.begin(), m.end(), expected, expected + 8));
          if (g_counter.throws == 1) {
            // We reached the first throw but not the second throw.
            break;
          }
        }
      }
    }
  }
  {
    using M = std::flat_map<int, ThrowingAssignment, ThrowingComparator>;
    for (int first_throw = 1; first_throw < 99; ++first_throw) {
      for (int second_throw = 1; second_throw < 99; ++second_throw) {
        g_counter = {0, 0, 0};
        M m       = M({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8});
        try {
          g_counter = {first_throw, second_throw, 0};
          auto n    = std::erase_if(m, ErasurePredicate());
          assert(n == 3);
          // If it didn't throw at all, we're done.
          g_counter = {0, 0, 0};
          assert((m == M{{1, 1}, {2, 2}, {6, 6}, {7, 7}, {8, 8}}));
          first_throw = 99; // "done"
          break;
        } catch (int ex) {
          assert(ex == 42);
          check_invariant(m);
          LIBCPP_ASSERT(m.empty() || std::equal(m.begin(), m.end(), expected, expected + 8));
          if (g_counter.throws == 1) {
            // We reached the first throw but not the second throw.
            break;
          }
        }
      }
    }
  }
  {
    using M =
        std::flat_map<ThrowingAssignment, int, ThrowingComparator, std::deque<ThrowingAssignment>, std::deque<int>>;
    for (int first_throw = 1; first_throw < 99; ++first_throw) {
      for (int second_throw = 1; second_throw < 99; ++second_throw) {
        g_counter                                = {0, 0, 0};
        std::deque<ThrowingAssignment> container = {5, 6, 7, 8};
        container.insert(container.begin(), {1, 2, 3, 4});
        M m = M(std::move(container), {1, 2, 3, 4, 5, 6, 7, 8});
        try {
          g_counter = {first_throw, second_throw, 0};
          auto n    = std::erase_if(m, ErasurePredicate());
          assert(n == 3);
          // If it didn't throw at all, we're done.
          g_counter = {0, 0, 0};
          assert((m == M{{1, 1}, {2, 2}, {6, 6}, {7, 7}, {8, 8}}));
          first_throw = 99; // "done"
          break;
        } catch (int ex) {
          assert(ex == 42);
          check_invariant(m);
          LIBCPP_ASSERT(m.empty() || std::equal(m.begin(), m.end(), expected, expected + 8));
          if (g_counter.throws == 1) {
            // We reached the first throw but not the second throw.
            break;
          }
        }
      }
    }
  }
  return 0;
}
