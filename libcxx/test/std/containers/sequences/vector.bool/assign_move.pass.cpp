//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>

// vector& operator=(vector&& c);

#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 bool tests() {
  //
  // Testing for O(1) ownership move
  //
  { // Test with pocma = true_type, thus performing an ownership move.
    std::vector<bool, other_allocator<bool> > l(other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > lo(other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > l2(other_allocator<bool>(6));
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    l2 = std::move(l);
    assert(l2 == lo);
    assert(l.empty());
    assert(l2.get_allocator() == lo.get_allocator());
  }
  { // Test with pocma = false_type and equal allocators, thus performing an ownership move.
    std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(5));
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(l.empty());
    assert(l2.get_allocator() == lo.get_allocator());
  }
  { // Test with pocma = false_type and equal allocators, thus performing an ownership move.
    std::vector<bool, min_allocator<bool> > l(min_allocator<bool>{});
    std::vector<bool, min_allocator<bool> > lo(min_allocator<bool>{});
    std::vector<bool, min_allocator<bool> > l2(min_allocator<bool>{});
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    l2 = std::move(l);
    assert(l2 == lo);
    assert(l.empty());
    assert(l2.get_allocator() == lo.get_allocator());
  }

  //
  // Testing for O(n) element-wise move
  //
  { // Test with pocma = false_type and unequal allocators, thus performing an element-wise move.
    // Reallocation occurs during the element-wise move due to insufficient space.
    std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(6));
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    l2 = std::move(l);
    assert(l2 == lo);
    assert(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(6));
  }

  { // Test with pocma = false_type and unequal allocators, thus performing an element-wise move.
    // No reallocation occurs since source data fits within current size.
    std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(6));
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    for (int i = 1; i <= 5; ++i)
      l2.push_back(i);
    l2 = std::move(l);
    assert(l2 == lo);
    assert(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(6));
  }
  { // Test with pocma = false_type and unequal allocators, thus performing an element-wise move.
    // No reallocation occurs since source data fits within current spare space.
    std::vector<bool, test_allocator<bool> > l(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(6));
    for (int i = 1; i <= 3; ++i) {
      l.push_back(i);
      lo.push_back(i);
    }
    for (int i = 1; i <= 2; ++i)
      l2.push_back(i);
    l2.reserve(5);
    l2 = std::move(l);
    assert(l2 == lo);
    assert(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(6));
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
