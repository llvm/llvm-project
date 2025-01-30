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

#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <unsigned N, class A>
TEST_CONSTEXPR_CXX20 void test(const A& a1, const A& a2) {
  std::vector<bool, A> l(N, true, a1);
  std::vector<bool, A> lo(N, true, a1);
  for (unsigned i = 1; i < N; i += 2) {
    l[i]  = false;
    lo[i] = false;
  }
  std::vector<bool, A> l2(std::move(l), a2);
  assert(l2 == lo);
  if (a1 == a2)
    assert(l.empty());
  else
    LIBCPP_ASSERT(!l.empty()); // For incompatible allocators, l is not guaranteed to be empty after the move.
  assert(l2.get_allocator() == a2);
}

TEST_CONSTEXPR_CXX20 bool tests() {
  { // Test with default allocator: compatible allocators
    using A = std::allocator<bool>;
    test<5>(A(), A());
    test<17>(A(), A());
    test<65>(A(), A());
    test<257>(A(), A());
  }

  { // Test with test_allocator: compatible and incompatible allocators
    using A = test_allocator<bool>;

    // Compatible allocators
    test<5>(A(5), A(5));
    test<17>(A(5), A(5));
    test<65>(A(5), A(5));
    test<257>(A(5), A(5));

    // Incompatible allocators
    test<5>(A(5), A(6));
    test<17>(A(5), A(6));
    test<65>(A(5), A(6));
    test<257>(A(5), A(6));
  }

  { // Test with other_allocator: compatible and incompatible allocators
    using A = other_allocator<bool>;

    // Compatible allocators
    test<5>(A(5), A(5));
    test<17>(A(5), A(5));
    test<65>(A(5), A(5));
    test<257>(A(5), A(5));

    // Incompatible allocators
    test<5>(A(5), A(3));
    test<17>(A(5), A(3));
    test<65>(A(5), A(3));
    test<257>(A(5), A(3));
  }

  { // Test with min_allocator: compatible allocators
    using A = min_allocator<bool>;
    test<5>(A(), A());
    test<17>(A(), A());
    test<65>(A(), A());
    test<257>(A(), A());
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
