//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// iterator erase(const_iterator first, const_iterator last);

#include <vector>
#include <cassert>
#include <memory>
#include <string>

#include "asan_testing.h"
#include "common.h"
#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_macros.h"

template <template <class> class Allocator, class T>
TEST_CONSTEXPR_CXX20 void tests() {
  {
    T arr[]             = {T(1), T(2), T(3)};
    using Vector        = std::vector<T, Allocator<T> >;
    using Iterator      = typename Vector::iterator;
    using ConstIterator = typename Vector::const_iterator;

    // Erase an empty range [first, last): last should be returned
    {
      {
        Vector v;
        Iterator i = v.erase(v.end(), v.end());
        assert(v.empty());
        assert(i == v.end());
        assert(is_contiguous_container_asan_correct(v));
      }
      {
        Vector v(arr, arr + 3);
        ConstIterator first = v.cbegin(), last = v.cbegin();
        Iterator i = v.erase(first, last);
        assert(v == Vector(arr, arr + 3));
        assert(i == last);
        assert(is_contiguous_container_asan_correct(v));
      }
      {
        Vector v(arr, arr + 3);
        ConstIterator first = v.cbegin() + 1, last = v.cbegin() + 1;
        Iterator i = v.erase(first, last);
        assert(v == Vector(arr, arr + 3));
        assert(i == last);
        assert(is_contiguous_container_asan_correct(v));
      }
      {
        Vector v(arr, arr + 3);
        ConstIterator first = v.cbegin(), last = v.cbegin();
        Iterator i = v.erase(first, last);
        assert(v == Vector(arr, arr + 3));
        assert(i == last);
        assert(is_contiguous_container_asan_correct(v));
      }
    }

    // Erase non-empty ranges
    {
      // Starting at begin()
      {
        {
          Vector v(arr, arr + 3);
          Iterator i = v.erase(v.cbegin(), v.cbegin() + 1);
          assert(v == Vector(arr + 1, arr + 3));
          assert(i == v.begin());
          assert(is_contiguous_container_asan_correct(v));
        }
        {
          Vector v(arr, arr + 3);
          Iterator i = v.erase(v.cbegin(), v.cbegin() + 2);
          assert(v == Vector(arr + 2, arr + 3));
          assert(i == v.begin());
          assert(is_contiguous_container_asan_correct(v));
        }
        {
          Vector v(arr, arr + 3);
          Iterator i = v.erase(v.cbegin(), v.end());
          assert(v.size() == 0);
          assert(i == v.begin());
          assert(is_contiguous_container_asan_correct(v));
        }
      }
      {
        Vector v(arr, arr + 3);
        Iterator i = v.erase(v.cbegin() + 1, v.cbegin() + 2);
        assert(v.size() == 2);
        assert(v[0] == arr[0]);
        assert(v[1] == arr[2]);
        assert(i == v.begin() + 1);
        assert(is_contiguous_container_asan_correct(v));
      }
      {
        Vector v(arr, arr + 3);
        Iterator i = v.erase(v.cbegin() + 1, v.cend());
        assert(v == Vector(arr, arr + 1));
        assert(i == v.begin() + 1);
        assert(is_contiguous_container_asan_correct(v));
      }
    }
  }
  {
    using InnerVector = std::vector<T, Allocator<T> >;
    using Vector      = std::vector<InnerVector, Allocator<InnerVector> >;
    Vector outer(2, InnerVector(1));
    outer.erase(outer.begin(), outer.begin());
    assert(outer.size() == 2);
    assert(outer[0].size() == 1);
    assert(outer[1].size() == 1);
    assert(is_contiguous_container_asan_correct(outer));
    assert(is_contiguous_container_asan_correct(outer[0]));
    assert(is_contiguous_container_asan_correct(outer[1]));
  }

  // Make sure vector::erase works with move-only types
  {
    // When non-trivial
    {
      std::vector<MoveOnly, Allocator<MoveOnly> > v;
      v.emplace_back(1);
      v.emplace_back(2);
      v.emplace_back(3);
      v.erase(v.begin(), v.begin() + 2);
      assert(v.size() == 1);
      assert(v[0] == MoveOnly(3));
    }
    // When trivial
    {
      std::vector<TrivialMoveOnly, Allocator<TrivialMoveOnly> > v;
      v.emplace_back(1);
      v.emplace_back(2);
      v.emplace_back(3);
      v.erase(v.begin(), v.begin() + 2);
      assert(v.size() == 1);
      assert(v[0] == TrivialMoveOnly(3));
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
#if TEST_STD_VER >= 20
  static_assert(tests());
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
  // Test for LWG2853:
  // Throws: Nothing unless an exception is thrown by the assignment operator or move assignment operator of T.
  {
    Throws arr[] = {1, 2, 3};
    std::vector<Throws> v(arr, arr + 3);
    Throws::sThrows = true;
    v.erase(v.begin(), --v.end());
    assert(v.size() == 1);
    v.erase(v.begin(), v.end());
    assert(v.size() == 0);
  }
#endif

  // Real world example with std::string, mostly intended to test trivial relocation
  {
    std::vector<std::string> v;

    // fill the vector with half short string and half long strings
    std::string short_string = "short";
    std::string long_string(256, 'x');
    for (int i = 0; i != 10; ++i) {
      v.push_back(i % 2 == 0 ? short_string : long_string);
    }

    std::vector<std::string> original = v;

    auto it = v.erase(v.cbegin() + 2, v.cbegin() + 8);
    assert(v.size() == 4);
    assert(v[0] == original[0]);
    assert(v[1] == original[1]);
    assert(v[2] == original[8]);
    assert(v[3] == original[9]);
    assert(it == v.begin() + 2);
  }

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

    // Erase elements [1] and [2] from it. Elements [3] [4] should be shifted, so we should
    // see 2 move assignments (and nothing else).
    v.erase(v.begin() + 1, v.begin() + 3);
    assert(v.size() == 3);
    assert(tracker.copy_assignments == 0);
    assert(tracker.move_assignments == 2);
  }
#endif

  return 0;
}
