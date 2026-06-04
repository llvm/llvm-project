//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <deque>

// constexpr since C++26

// deque& operator=(initializer_list<value_type> il);

#include "asan_testing.h"
#include <deque>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"

#if TEST_STD_VER >= 26
constexpr bool test() {
  std::deque<int> d;
  d = {1, 2, 3};
  assert((d == std::deque<int>{1, 2, 3}));
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  test();
  static_assert(test());
#endif

  {
    std::deque<int> d;
    d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d));
  }
  {
    std::deque<int, min_allocator<int>> d;
    d = {3, 4, 5, 6};
    assert(d.size() == 4);
    assert(d[0] == 3);
    assert(d[1] == 4);
    assert(d[2] == 5);
    assert(d[3] == 6);
    LIBCPP_ASSERT(is_double_ended_contiguous_container_asan_correct(d));
  }

  return 0;
}
