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

// vector(vector&& c, const allocator_type& a);

#include <array>
#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <unsigned N, class A>
TEST_CONSTEXPR_CXX20 void test(const A& a, const A& a0) {
  std::vector<bool, A> v(N, false, a);
  std::vector<bool, A> original(N, false, a0);
  for (unsigned i = 1; i < N; i += 2) {
    v[i]        = true;
    original[i] = true;
  }
  std::vector<bool, A> v2(std::move(v), a0);
  assert(v2 == original);
  assert(v2.get_allocator() == a0);
  if (a == a0)
    assert(v.empty()); // After container-move, the vector is guaranteed to be empty
  else
    LIBCPP_ASSERT(!v.empty()); // After element-wise move, the RHS vector is not necessarily empty
}

TEST_CONSTEXPR_CXX20 bool tests() {
  { // Test with default allocator: compatible allocators
    using A = std::allocator<bool>;
    test<5>(A(), A());
    test<17>(A(), A());
    test<65>(A(), A());
    test<299>(A(), A());
  }
  { // Test with test_allocator: compatible and incompatible allocators
    using A = test_allocator<bool>;

    // Compatible allocators
    test<5>(A(5), A(5));
    test<17>(A(5), A(5));
    test<65>(A(5), A(5));
    test<299>(A(5), A(5));

    // Incompatible allocators
    test<5>(A(5), A(6));
    test<17>(A(5), A(6));
    test<65>(A(5), A(6));
    test<299>(A(5), A(6));
  }
  { // Test with other_allocator: compatible and incompatible allocators
    using A = other_allocator<bool>;

    // Compatible allocators
    test<5>(A(5), A(5));
    test<17>(A(5), A(5));
    test<65>(A(5), A(5));
    test<299>(A(5), A(5));

    // Incompatible allocators
    test<5>(A(5), A(3));
    test<17>(A(5), A(3));
    test<65>(A(5), A(3));
    test<299>(A(5), A(3));
  }
  { // Test with min_allocator: compatible allocators
    using A = min_allocator<bool>;
    test<5>(A(), A());
    test<17>(A(), A());
    test<65>(A(), A());
    test<299>(A(), A());
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
