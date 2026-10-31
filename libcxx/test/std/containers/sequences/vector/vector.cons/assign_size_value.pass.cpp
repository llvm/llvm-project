//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// void assign(size_type n, const_reference v);

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

#include <vector>
#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"
#include "../common.h"
#include "count_new.h"

TEST_CONSTEXPR bool is6(int x) { return x == 6; }

template <typename Vec>
TEST_CONSTEXPR_CXX20 void test(Vec& v) {
  v.assign(5, 6);
  assert(v.size() == 5);
  assert(is_contiguous_container_asan_correct(v));
  assert(std::all_of(v.begin(), v.end(), is6));
}

TEST_CONSTEXPR_CXX20 bool tests() {
  {
    typedef std::vector<int> V;
    V d1;
    V d2;
    d2.reserve(10); // no reallocation during assign.
    test(d1);
    test(d2);
  }
  {
    std::vector<int> vec;
    vec.reserve(32);
    vec.resize(16); // destruction during assign
    test(vec);
  }
#if TEST_STD_VER >= 11
  {
    typedef std::vector<int, min_allocator<int>> V;
    V d1;
    V d2;
    d2.reserve(10); // no reallocation during assign.
    test(d1);
    test(d2);
  }
#endif

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif

  // Verify that assign(n, x) destroys partially-constructed elements when a
  // later element's copy constructor throws.
#if !defined(TEST_HAS_NO_EXCEPTIONS)
  {
    // Without reallocation (note the reserve call)
    int throw_after = 4; // x consumes 1, then 3 successful copies, 4th throws
    std::vector<throwing_t> v;
    v.reserve(10);
    try {
      throwing_t x(throw_after);
      v.assign(5, x);
      assert(false);
    } catch (int) {
      LIBCPP_ASSERT(v.size() == 0);
    }
  }
  check_new_delete_called();
  {
    // With reallocation
    int throw_after = 4; // x consumes 1, then 3 successful copies, 4th throws
    std::vector<throwing_t> v;
    try {
      throwing_t x(throw_after);
      v.assign(5, x);
      assert(false);
    } catch (int) {
      LIBCPP_ASSERT(v.size() == 0);
    }
  }
  check_new_delete_called();
#endif
  return 0;
}
