//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector(const vector& v);

#include <array>
#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class C>
TEST_CONSTEXPR_CXX20 void test(const C& x) {
  typename C::size_type s = x.size();
  C c(x);
  LIBCPP_ASSERT(c.__invariants());
  assert(c.size() == s);
  assert(c == x);
#if TEST_STD_VER >= 11
  assert(c.get_allocator() ==
         std::allocator_traits<typename C::allocator_type>::select_on_container_copy_construction(x.get_allocator()));
#endif
}

TEST_CONSTEXPR_CXX20 bool tests() {
  std::array<int, 5> a1   = {1, 0, 1, 0, 1};
  std::array<int, 18> a2  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 33> a3  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 65> a4  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 299> a5 = {};
  for (unsigned i = 0; i < a5.size(); i += 2)
    a5[i] = 1;

  // Tests for vector<bool> copy constructor with word size up to 5 (i.e., bit size > 256 on a 64-bit system)
  { // Test with default std::allocator
    test(std::vector<bool>(a1.begin(), a1.end()));
    test(std::vector<bool>(a2.begin(), a2.end()));
    test(std::vector<bool>(a3.begin(), a3.end()));
    test(std::vector<bool>(a4.begin(), a4.end()));
    test(std::vector<bool>(a5.begin(), a5.end()));
  }
  { // Test with test_allocator
    using A = test_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end()));
    test(C(a2.begin(), a2.end()));
    test(C(a3.begin(), a3.end()));
    test(C(a4.begin(), a4.end()));
    test(C(a5.begin(), a5.end()));
  }
  { // Test with other_allocator
    using A = other_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end()));
    test(C(a2.begin(), a2.end()));
    test(C(a3.begin(), a3.end()));
    test(C(a4.begin(), a4.end()));
    test(C(a5.begin(), a5.end()));
  }
  { // Test with min_allocator
    using A = min_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end()));
    test(C(a2.begin(), a2.end()));
    test(C(a3.begin(), a3.end()));
    test(C(a4.begin(), a4.end()));
    test(C(a5.begin(), a5.end()));
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
