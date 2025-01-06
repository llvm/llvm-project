//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator erase(const_iterator position);

#include <vector>
#include <cassert>
#include <memory>

#include "asan_testing.h"
#include "common.h"
#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_macros.h"

template <template <class> class Allocator, class T>
TEST_CONSTEXPR_CXX20 void tests() {
  {
    T arr[]             = {T(1), T(2), T(3), T(4), T(5)};
    using Vector        = std::vector<T, Allocator<T> >;
    using Iterator      = typename Vector::iterator;

    {
      Vector v(arr, arr + 5);
      Iterator it = v.erase(v.cbegin());
      assert(v == Vector(arr + 1, arr + 5));
      assert(it == v.begin());
      assert(is_contiguous_container_asan_correct(v));
    }
    {
      T expected[] = {T(1), T(3), T(4), T(5)};
      Vector v(arr, arr + 5);
      Iterator it = v.erase(v.cbegin() + 1);
      assert(v == Vector(expected, expected + 4));
      assert(it == v.begin() + 1);
      assert(is_contiguous_container_asan_correct(v));
    }
    {
      T expected[] = {T(1), T(2), T(3), T(4)};
      Vector v(arr, arr + 5);
      Iterator it = v.erase(v.cbegin() + 4);
      assert(v == Vector(expected, expected + 4));
      assert(it == v.end());
      assert(is_contiguous_container_asan_correct(v));
    }
  }

  // Make sure vector::erase works with move-only types
  {
    // When non-trivial
    {
      std::vector<MoveOnly, Allocator<MoveOnly> > v;
      v.emplace_back(1);
      v.emplace_back(2);
      v.emplace_back(3);
      v.erase(v.begin());
      assert(v.size() == 2);
      assert(v[0] == MoveOnly(2));
      assert(v[1] == MoveOnly(3));
    }
    // When trivial
    {
      std::vector<TrivialMoveOnly, Allocator<TrivialMoveOnly> > v;
      v.emplace_back(1);
      v.emplace_back(2);
      v.emplace_back(3);
      v.erase(v.begin());
      assert(v.size() == 2);
      assert(v[0] == TrivialMoveOnly(2));
      assert(v[1] == TrivialMoveOnly(3));
    }
  }
}

TEST_CONSTEXPR_CXX20 bool tests() {
  tests<std::allocator, int>();
  tests<std::allocator, NonTriviallyRelocatable>();
  tests<min_allocator, int>();
  tests<min_allocator, NonTriviallyRelocatable>();
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
  // Test for LWG2853:
  // Throws: Nothing unless an exception is thrown by the assignment operator or move assignment operator of T.
  {
    Throws arr[] = {1, 2, 3};
    std::vector<Throws> v(arr, arr + 3);
    Throws::sThrows = true;
    v.erase(v.begin());
    v.erase(--v.end());
    v.erase(v.begin());
    assert(v.size() == 0);
  }
#endif

  // Make sure we satisfy the complexity requirement in terms of the number of times the assignment
  // operator is called.
  //
  // There is currently ambiguity as to whether this is truly mandated by the Standard, so we only
  // test it for libc++.
#ifdef _LIBCPP_VERSION
  {
    Tracker tracker;
    std::vector<TrackedAssignment> v;

    // Set up the vector with 5 elements.
    for (int i = 0; i != 5; ++i) {
      v.emplace_back(&tracker);
    }
    assert(tracker.copy_assignments == 0);
    assert(tracker.move_assignments == 0);

    // Erase element [1] from it. Elements [2] [3] [4] should be shifted, so we should
    // see 3 move assignments (and nothing else).
    v.erase(v.begin() + 1);
    assert(v.size() == 4);
    assert(tracker.copy_assignments == 0);
    assert(tracker.move_assignments == 3);
  }
#endif

  return 0;
}
