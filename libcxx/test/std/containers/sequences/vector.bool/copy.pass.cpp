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

#include <cassert>
#include <memory>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class A>
TEST_CONSTEXPR_CXX20 void test(const std::vector<bool, A>& x) {
  std::vector<bool, A> c(x);
  LIBCPP_ASSERT(c.__invariants());
  assert(c.size() == x.size());
  assert(c == x);
#if TEST_STD_VER >= 11
  assert(c.get_allocator() == std::allocator_traits<A>::select_on_container_copy_construction(x.get_allocator()));
#endif
}

TEST_CONSTEXPR_CXX20 bool tests() {
  bool a05[5]  = {1, 0, 1, 0, 1};
  bool a17[17] = {0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1};
  bool a33[33] = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
  bool a65[65] = {0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};

  { // Test the copy constructor with the default allocator
    test(std::vector<bool>(a05, a05 + sizeof(a05) / sizeof(a05[0])));
    test(std::vector<bool>(a17, a17 + sizeof(a17) / sizeof(a17[0])));
    test(std::vector<bool>(a33, a33 + sizeof(a33) / sizeof(a33[0])));
    test(std::vector<bool>(a65, a65 + sizeof(a65) / sizeof(a65[0])));
    test(std::vector<bool>(257, true));
  }

  { // Test the copy constructor with test_allocator
    using A = test_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a05, a05 + sizeof(a05) / sizeof(a05[0]), A(5)));
    test(C(a17, a17 + sizeof(a17) / sizeof(a17[0]), A(5)));
    test(C(a33, a33 + sizeof(a33) / sizeof(a33[0]), A(5)));
    test(C(a65, a65 + sizeof(a65) / sizeof(a65[0]), A(5)));
    test(C(257, true, A(5)));
  }

  { // Test the copy constructor with other_allocator
    using A = other_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a05, a05 + sizeof(a05) / sizeof(a05[0]), A(5)));
    test(C(a17, a17 + sizeof(a17) / sizeof(a17[0]), A(5)));
    test(C(a33, a33 + sizeof(a33) / sizeof(a33[0]), A(5)));
    test(C(a65, a65 + sizeof(a65) / sizeof(a65[0]), A(5)));
    test(C(257, true, A(5)));
  }

  { // Test the copy constructor with min_allocator
    using A = min_allocator<bool>;
    using C = std::vector<bool, A>;
    test(C(a05, a05 + sizeof(a05) / sizeof(a05[0]), A()));
    test(C(a17, a17 + sizeof(a17) / sizeof(a17[0]), A()));
    test(C(a33, a33 + sizeof(a33) / sizeof(a33[0]), A()));
    test(C(a65, a65 + sizeof(a65) / sizeof(a65[0]), A()));
    test(C(257, true, A()));
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif
  return 0;
}
