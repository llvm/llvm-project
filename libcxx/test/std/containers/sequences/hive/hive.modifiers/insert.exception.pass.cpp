//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: libcpp-has-no-experimental-hive

// <hive>

// UNSUPPORTED: no-exceptions

// emplace and insert: if an exception is thrown, there are no effects. Size, capacity, the elements and the
// allocations are unchanged.

#include <algorithm>
#include <cassert>
#include <climits>
#include <hive>
#include <memory>
#include <vector>

#include "../helpers.h"
#include "test_allocator.h"
#include "test_macros.h"

struct ThrowingCopy {
  static bool throw_next;
  int v;
  explicit ThrowingCopy(int x) : v(x) {}
  ThrowingCopy(const ThrowingCopy& o) : v(o.v) {
    if (throw_next) {
      throw_next = false;
      throw 42;
    }
  }
};
bool ThrowingCopy::throw_next = false;

template <class H>
void expect_throw(H& h, const ThrowingCopy& x) {
  try {
    h.insert(x);
    assert(false);
  } catch (int) {
  }
}

// The copy throws into an empty hive: no block is kept.
void test_throw_into_empty_hive() {
  test_allocator_statistics stats;
  std::hive<ThrowingCopy, test_allocator<ThrowingCopy>> h{test_allocator<ThrowingCopy>(&stats)};
  const ThrowingCopy x(1);
  ThrowingCopy::throw_next = true;
  expect_throw(h, x);
  assert(h.empty());
  assert(h.size() == 0);
  assert(h.capacity() == 0);
  assert(stats.alloc_count == 0);
  assert(h.begin() == h.end());
  check_invariant(h);

  h.insert(x);
  assert(h.size() == 1);
  assert(h.begin()->v == 1);
}

// The copy throws into a hive with elements: they, the capacity and the allocations are untouched.
void test_throw_into_filled_hive() {
  test_allocator_statistics stats;
  std::hive<ThrowingCopy, test_allocator<ThrowingCopy>> h{test_allocator<ThrowingCopy>(&stats)};
  for (int i = 0; i != 5; ++i) {
    h.insert(ThrowingCopy(i));
  }
  const auto capacity = h.capacity();
  const int allocs    = stats.alloc_count;
  const auto first    = h.begin();

  const ThrowingCopy x(9);
  ThrowingCopy::throw_next = true;
  expect_throw(h, x);
  assert(h.size() == 5);
  assert(h.capacity() == capacity);
  assert(stats.alloc_count == allocs);
  assert(h.begin() == first);
  std::vector<int> seen;
  for (const ThrowingCopy& e : h) {
    seen.push_back(e.v);
  }
  std::sort(seen.begin(), seen.end());
  assert((seen == std::vector<int>{0, 1, 2, 3, 4}));
  check_invariant(h);
}

// Each allocation the first insertion makes fails in turn: nothing is kept, and the hive is usable afterwards.
void test_allocation_throws() {
  int failed_allocations = 0;
  for (int throw_after = 0;; ++throw_after) {
    test_allocator_statistics stats;
    std::hive<int, test_allocator<int>> h{test_allocator<int>(&stats)};
    stats.throw_after = throw_after;
    bool threw        = false;
    try {
      h.insert(1);
    } catch (...) {
      threw = true;
    }
    if (!threw) {
      assert(h.size() == 1);
      break;
    }

    ++failed_allocations;
    assert(h.empty());
    assert(h.capacity() == 0);
    assert(stats.alloc_count == 0);
    assert(h.begin() == h.end());
    check_invariant(h);

    stats.throw_after = INT_MAX;
    h.insert(1);
    assert(h.size() == 1);
  }

  assert(failed_allocations > 0);
  // libc++ allocates a block as its slots and its header.
  LIBCPP_ASSERT(failed_allocations == 2);
}

int main(int, char**) {
  test_throw_into_empty_hive();
  test_throw_into_filled_hive();
  test_allocation_throws();

  return 0;
}
