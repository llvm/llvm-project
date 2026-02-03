//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <vector>
// vector<bool>

// vector& operator=(vector&& c);

#include <cassert>
#include <utility>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 void test_move_assignment(unsigned N) {
  //
  // Testing for container move where either POCMA = true_type or the allocators compare equal
  //
  { // Test with POCMA = true_type
    std::vector<bool, other_allocator<bool> > l(N, true, other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > lo(N, true, other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > l2(N + 10, false, other_allocator<bool>(42));
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(l.empty()); // After move, source vector is in a vliad but unspecified state. libc++ leaves it empty.
    assert(l2.get_allocator() == lo.get_allocator());
  }
  { // Test with POCMA = false_type and allocators compare equal
    std::vector<bool, test_allocator<bool> > l(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(N + 10, false, test_allocator<bool>(5));
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(l.empty());
    assert(l2.get_allocator() == lo.get_allocator());
  }
  { // Test with POCMA = false_type and allocators compare equal
    std::vector<bool, min_allocator<bool> > l(N, true, min_allocator<bool>{});
    std::vector<bool, min_allocator<bool> > lo(N, true, min_allocator<bool>{});
    std::vector<bool, min_allocator<bool> > l2(N + 10, false, min_allocator<bool>{});
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(l.empty());
    assert(l2.get_allocator() == lo.get_allocator());
  }

  //
  // Testing for element-wise move where POCMA = false_type and allocators compare unequal
  //
  { // Test with reallocation during the element-wise move due to empty destination vector.
    std::vector<bool, test_allocator<bool> > l(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(test_allocator<bool>(42));
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(42));
  }
  { // Test with reallocation occurs during the element-wise move due to insufficient destination space.
    std::vector<bool, test_allocator<bool> > l(N + 64, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(N + 64, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(10, false, test_allocator<bool>(42));
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(42));
  }
  { // Test without reallocation where source vector elements fit within destination size.
    std::vector<bool, test_allocator<bool> > l(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > lo(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(N * 2, false, test_allocator<bool>(42));
    l2 = std::move(l);
    assert(l2 == lo);
    LIBCPP_ASSERT(!l.empty());
    assert(l2.get_allocator() == test_allocator<bool>(42));
  }
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_move_assignment(3);
  test_move_assignment(18);
  test_move_assignment(33);
  test_move_assignment(65);
  test_move_assignment(299);

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
