//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template <class _Comp> struct __debug_three_way_comp

// Make sure __debug_three_way_comp asserts when the comparator is not consistent.

// UNSUPPORTED: !libcpp-has-debug-mode, c++03, c++11, c++14, c++17

#include <algorithm>
#include <iterator>

#include "check_assertion.h"

struct AlwaysLess {
  std::strong_ordering operator()(int, int) const { return std::strong_ordering::less; }
};

struct AlwaysGreater {
  std::strong_ordering operator()(int, int) const { return std::strong_ordering::greater; }
};

struct InconsistentEquals {
  std::strong_ordering operator()(int a, int) const {
    if (a == 0)
      return std::strong_ordering::equal;
    return std::strong_ordering::greater;
  }
};

int main(int, char**) {
  int zero = 0;
  int one  = 1;

  AlwaysLess alwaysLess;
  std::__debug_three_way_comp<AlwaysLess> debugAlwaysLess(alwaysLess);
  TEST_LIBCPP_ASSERT_FAILURE(debugAlwaysLess(zero, one), "Comparator does not induce a strict weak ordering");

  AlwaysGreater alwaysGreater;
  std::__debug_three_way_comp<AlwaysGreater> debugAlwaysGreater(alwaysGreater);
  TEST_LIBCPP_ASSERT_FAILURE(debugAlwaysGreater(zero, one), "Comparator does not induce a strict weak ordering");

  InconsistentEquals inconsistentEquals;
  std::__debug_three_way_comp<InconsistentEquals> debugInconsistentEquals(inconsistentEquals);
  TEST_LIBCPP_ASSERT_FAILURE(debugInconsistentEquals(zero, one), "Comparator does not induce a strict weak ordering");

  return 0;
}
