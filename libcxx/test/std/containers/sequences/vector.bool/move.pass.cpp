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

// vector(vector&& c);

#include <array>
#include <cassert>
#include <vector>

#include "min_allocator.h"
#include "test_allocator.h"
#include "test_macros.h"

template <unsigned N, class A>
TEST_CONSTEXPR_CXX20 void test(const A& a) {
  std::vector<bool, A> v(N, false, a);
  std::vector<bool, A> original(N, false, a);
  for (unsigned i = 1; i < N; i += 2) {
    v[i]        = true;
    original[i] = true;
  }
  std::vector<bool, A> v2 = std::move(v);
  assert(v2 == original);
  assert(v.empty()); // The moved-from vector is guaranteed to be empty after move-construction
  assert(v2.get_allocator() == original.get_allocator());
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test_allocator_statistics alloc_stats;

  // Tests for move constructor with word size up to 5 (i.e., bit size > 256 for 64-bit system)
  {
    using A = std::allocator<bool>;
    test<5>(A());
    test<18>(A());
    test<33>(A());
    test<65>(A());
    test<299>(A());
  }
  {
    using A = other_allocator<bool>;
    test<5>(A(5));
    test<18>(A(5));
    test<33>(A(5));
    test<65>(A(5));
    test<299>(A(5));
  }
  {
    using A = min_allocator<bool>;
    test<5>(A());
    test<18>(A());
    test<33>(A());
    test<65>(A());
    test<299>(A());
  }
  {
    using A = test_allocator<bool>;
    test<5>(A(5, &alloc_stats));
    test<18>(A(5, &alloc_stats));
    test<33>(A(5, &alloc_stats));
    test<65>(A(5, &alloc_stats));
    test<299>(A(5, &alloc_stats));
  }

  { // Tests to verify the allocator statistics after move
    alloc_stats.clear();
    using Vect   = std::vector<bool, test_allocator<bool> >;
    using AllocT = Vect::allocator_type;
    Vect v(test_allocator<bool>(42, 101, &alloc_stats));
    assert(alloc_stats.count == 1);
    {
      const AllocT& a = v.get_allocator();
      assert(alloc_stats.count == 2);
      assert(a.get_data() == 42);
      assert(a.get_id() == 101);
    }
    assert(alloc_stats.count == 1);
    alloc_stats.clear_ctor_counters();

    Vect v2 = std::move(v);
    assert(alloc_stats.count == 2);
    assert(alloc_stats.copied == 0);
    assert(alloc_stats.moved == 1);
    {
      const AllocT& a1 = v.get_allocator();
      assert(a1.get_id() == test_alloc_base::moved_value);
      assert(a1.get_data() == 42);

      const AllocT& a2 = v2.get_allocator();
      assert(a2.get_id() == 101);
      assert(a2.get_data() == 42);
      assert(a1 == a2);
    }
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
