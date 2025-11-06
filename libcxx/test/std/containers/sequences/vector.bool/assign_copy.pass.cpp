//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector& operator=(const vector& c);

#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

TEST_CONSTEXPR_CXX20 void test_copy_assignment(unsigned N) {
  //
  // Test with insufficient space where reallocation occurs during assignment
  //
  { // POCCA = true_type, thus copy-assign the allocator
    std::vector<bool, other_allocator<bool> > l(N, true, other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > l2(other_allocator<bool>(3));
    l2 = l;
    assert(l2 == l);
    assert(l2.get_allocator() == other_allocator<bool>(5));
  }
  { // POCCA = false_type, thus allocator is unchanged
    std::vector<bool, test_allocator<bool> > l(N + 64, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(10, false, test_allocator<bool>(3));
    l2 = l;
    assert(l2 == l);
    assert(l2.get_allocator() == test_allocator<bool>(3));
  }
  { // Stateless allocator
    std::vector<bool, min_allocator<bool> > l(N + 64, true, min_allocator<bool>());
    std::vector<bool, min_allocator<bool> > l2(N / 2, false, min_allocator<bool>());
    l2 = l;
    assert(l2 == l);
    assert(l2.get_allocator() == min_allocator<bool>());
  }

  //
  // Test with sufficient size where no reallocation occurs during assignment
  //
  { // POCCA = false_type, thus allocator is unchanged
    std::vector<bool, test_allocator<bool> > l(N, true, test_allocator<bool>(5));
    std::vector<bool, test_allocator<bool> > l2(N + 64, false, test_allocator<bool>(3));
    l2 = l;
    assert(l2 == l);
    assert(l2.get_allocator() == test_allocator<bool>(3));
  }
  { // POCCA = true_type, thus copy-assign the allocator
    std::vector<bool, other_allocator<bool> > l(N, true, other_allocator<bool>(5));
    std::vector<bool, other_allocator<bool> > l2(N * 2, false, other_allocator<bool>(3));
    l2.reserve(5);
    l2 = l;
    assert(l2 == l);
    assert(l2.get_allocator() == other_allocator<bool>(5));
  }
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_copy_assignment(3);
  test_copy_assignment(18);
  test_copy_assignment(33);
  test_copy_assignment(65);
  test_copy_assignment(299);

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
