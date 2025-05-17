//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>
// vector<bool>

// vector(const vector& v, const allocator_type& a);

#include <array>
#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class C>
TEST_CONSTEXPR_CXX20 void test(const C& x, const typename C::allocator_type& a) {
  typename C::size_type s = x.size();
  C c(x, a);
  LIBCPP_ASSERT(c.__invariants());
  assert(c.size() == s);
  assert(c == x);
  assert(c.get_allocator() == a);
}

TEST_CONSTEXPR_CXX20 bool tests() {
  std::array<int, 5> a1   = {1, 0, 1, 0, 1};
  std::array<int, 18> a2  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 33> a3  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 65> a4  = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0};
  std::array<int, 299> a5 = {};
  for (unsigned i = 0; i < a5.size(); i += 2)
    a5[i] = 1;

  // Tests for allocator-extended copy constructor with word size up to 5 (i.e., bit size > 256 on a 64-bit system)
  { // Test with the default std::allocator
    test(std::vector<bool>(a1.begin(), a1.end()), std::allocator<bool>());
    test(std::vector<bool>(a2.begin(), a2.end()), std::allocator<bool>());
    test(std::vector<bool>(a3.begin(), a3.end()), std::allocator<bool>());
    test(std::vector<bool>(a4.begin(), a4.end()), std::allocator<bool>());
    test(std::vector<bool>(a5.begin(), a5.end()), std::allocator<bool>());
  }
  { // Test with test_allocator
    using A = test_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end(), A(5)), A(3));
    test(C(a2.begin(), a2.end(), A(5)), A(3));
    test(C(a3.begin(), a3.end(), A(5)), A(3));
    test(C(a4.begin(), a4.end(), A(5)), A(3));
    test(C(a5.begin(), a5.end(), A(5)), A(3));
  }
  { // Test with other_allocator
    using A = other_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end(), A(5)), A(3));
    test(C(a2.begin(), a2.end(), A(5)), A(3));
    test(C(a3.begin(), a3.end(), A(5)), A(3));
    test(C(a4.begin(), a4.end(), A(5)), A(3));
    test(C(a5.begin(), a5.end(), A(5)), A(3));
  }
  { // Test with min_allocator
    using A = min_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a1.begin(), a1.end(), A()), A());
    test(C(a2.begin(), a2.end(), A()), A());
    test(C(a3.begin(), a3.end(), A()), A());
    test(C(a4.begin(), a4.end(), A()), A());
    test(C(a5.begin(), a5.end(), A()), A());
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
