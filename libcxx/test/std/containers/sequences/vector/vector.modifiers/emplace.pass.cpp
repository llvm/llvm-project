//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03 && !stdlib=libc++
// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=9000000

// <vector>

// template <class... Args> iterator emplace(const_iterator pos, Args&&... args);

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include "asan_testing.h"
#include "common.h"
#include "min_allocator.h"
#include "MoveOnly.h"
#include "test_allocator.h"
#include "test_macros.h"

template <class T>
struct has_moved_from_sentinel_value : std::false_type {};

template <>
struct has_moved_from_sentinel_value<MoveOnly> : std::true_type {};

template <template <class...> class Allocator, class T>
TEST_CONSTEXPR_CXX20 void test() {
  using Vector   = std::vector<T, Allocator<T> >;
  using Iterator = typename Vector::iterator;

  // Check the return type
  {
    Vector v;
    ASSERT_SAME_TYPE(decltype(v.emplace(v.cbegin(), 1)), Iterator);
  }

  // Emplace at the end of a vector with increasing size
  {
    Vector v;

    // starts with size 0
    {
      Iterator it = v.emplace(v.cend(), 0);
      assert(it == v.end() - 1);
      assert(v.size() == 1);
      assert(v[0] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size 1
    {
      Iterator it = v.emplace(v.cend(), 1);
      assert(it == v.end() - 1);
      assert(v.size() == 2);
      assert(v[0] == T(0));
      assert(v[1] == T(1));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size 2
    {
      Iterator it = v.emplace(v.cend(), 2);
      assert(it == v.end() - 1);
      assert(v.size() == 3);
      assert(v[0] == T(0));
      assert(v[1] == T(1));
      assert(v[2] == T(2));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size n...
    for (std::size_t n = 3; n != 100; ++n) {
      Iterator it = v.emplace(v.cend(), n);
      assert(it == v.end() - 1);
      assert(v.size() == n + 1);
      for (std::size_t i = 0; i != n + 1; ++i)
        assert(v[i] == T(i));
      assert(is_contiguous_container_asan_correct(v));
    }
  }

  // Emplace at the start of a vector with increasing size
  {
    Vector v;

    // starts with size 0
    {
      Iterator it = v.emplace(v.cbegin(), 0);
      assert(it == v.begin());
      assert(v.size() == 1);
      assert(v[0] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size 1
    {
      Iterator it = v.emplace(v.cbegin(), 1);
      assert(it == v.begin());
      assert(v.size() == 2);
      assert(v[0] == T(1));
      assert(v[1] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size 2
    {
      Iterator it = v.emplace(v.cbegin(), 2);
      assert(it == v.begin());
      assert(v.size() == 3);
      assert(v[0] == T(2));
      assert(v[1] == T(1));
      assert(v[2] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }

    // starts with size n...
    for (std::size_t n = 3; n != 100; ++n) {
      Iterator it = v.emplace(v.cbegin(), n);
      assert(it == v.begin());
      assert(v.size() == n + 1);
      for (std::size_t i = 0; i != n + 1; ++i)
        assert(v[i] == T(n - i));
      assert(is_contiguous_container_asan_correct(v));
    }
  }

  // Emplace somewhere inside the vector
  {
    Vector v;
    v.emplace_back(0);
    v.emplace_back(1);
    v.emplace_back(2);
    // vector is {0, 1, 2}

    {
      Iterator it = v.emplace(v.cbegin() + 1, 3);
      // vector is {0, 3, 1, 2}
      assert(it == v.begin() + 1);
      assert(v.size() == 4);
      assert(v[0] == T(0));
      assert(v[1] == T(3));
      assert(v[2] == T(1));
      assert(v[3] == T(2));
      assert(is_contiguous_container_asan_correct(v));
    }

    {
      Iterator it = v.emplace(v.cbegin() + 2, 4);
      // vector is {0, 3, 4, 1, 2}
      assert(it == v.begin() + 2);
      assert(v.size() == 5);
      assert(v[0] == T(0));
      assert(v[1] == T(3));
      assert(v[2] == T(4));
      assert(v[3] == T(1));
      assert(v[4] == T(2));
      assert(is_contiguous_container_asan_correct(v));
    }
  }

  // Emplace after reserving
  {
    Vector v;
    v.emplace_back(0);
    v.emplace_back(1);
    v.emplace_back(2);
    // vector is {0, 1, 2}

    v.reserve(1000);
    Iterator it = v.emplace(v.cbegin() + 1, 3);
    assert(it == v.begin() + 1);
    assert(v.size() == 4);
    assert(v[0] == T(0));
    assert(v[1] == T(3));
    assert(v[2] == T(1));
    assert(v[3] == T(2));
    assert(is_contiguous_container_asan_correct(v));
  }

  // Emplace with the same type that's stored in the vector (as opposed to just constructor arguments)
  {
    Vector v;
    Iterator it = v.emplace(v.cbegin(), T(1));
    assert(it == v.begin());
    assert(v.size() == 1);
    assert(v[0] == T(1));
    assert(is_contiguous_container_asan_correct(v));
  }

  // Emplace from an element inside the vector itself. This is interesting for two reasons. First, if the
  // vector must increase capacity, the implementation needs to make sure that it doesn't end up inserting
  // from a dangling reference.
  //
  // Second, if the vector doesn't need to grow but its elements get shifted internally, the implementation
  // must make sure that it doesn't end up inserting from an element whose position has changed.
  {
    // When capacity must increase
    {
      Vector v;
      v.emplace_back(1);
      v.emplace_back(2);

      while (v.size() < v.capacity()) {
        v.emplace_back(3);
      }
      assert(v.size() == v.capacity());
      // vector is {1, 2, 3...}

      std::size_t old_cap = v.capacity();
      v.emplace(v.cbegin(), std::move(v[1]));
      assert(v.capacity() > old_cap); // test the test

      // vector is {2, 1, 0, 3...}
      // Note that old v[1] has been set to 0 when it was moved-from
      assert(v.size() >= 3);
      assert(v[0] == T(2));
      assert(v[1] == T(1));
      if (has_moved_from_sentinel_value<T>::value)
        assert(v[2] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }

    // When elements shift around
    {
      Vector v;
      v.emplace_back(1);
      v.emplace_back(2);
      // vector is {1, 2}

      v.reserve(3);
      std::size_t old_cap = v.capacity();
      v.emplace(v.cbegin(), std::move(v[1]));
      assert(v.capacity() == old_cap); // test the test

      // vector is {2, 1, 0}
      // Note that old v[1] has been set to 0 when it was moved-from
      assert(v.size() == 3);
      assert(v[0] == T(2));
      assert(v[1] == T(1));
      if (has_moved_from_sentinel_value<T>::value)
        assert(v[2] == T(0));
      assert(is_contiguous_container_asan_correct(v));
    }
  }

  // Make sure that we don't reallocate when we have sufficient capacity
  {
    Vector v;
    v.reserve(8);
    assert(v.capacity() >= 8);

    std::size_t old_capacity = v.capacity();
    v.emplace_back(0);
    v.emplace_back(1);
    v.emplace_back(2);
    v.emplace_back(3);
    assert(v.capacity() == old_capacity);

    v.emplace(v.cend(), 4);
    assert(v.size() == 5);
    assert(v.capacity() == old_capacity);
    assert(v[0] == T(0));
    assert(v[1] == T(1));
    assert(v[2] == T(2));
    assert(v[3] == T(3));
    assert(v[4] == T(4));
    assert(is_contiguous_container_asan_correct(v));
  }

  // Make sure that we correctly handle the case where an exception would be thrown if moving the element into place.
  // This is a very specific test that aims to validate that the implementation doesn't create a temporary object e.g.
  // on the stack and then moves it into its final location inside the newly allocated vector storage.
  //
  // If that were the case, and if the element happened to throw upon move construction or move assignment into its
  // final location, we would have invalidated iterators, when a different approach would allow us to still provide
  // the strong exception safety guarantee.
  //
  // Instead of the naive approach, libc++ emplaces the new element into its final location immediately, and only
  // after this has been done do we start making non-reversible changes to the vector's underlying storage. This
  // test pins down that behavior, although that is something that we don't advertise widely and could potentially
  // change in the future.
#if defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_EXCEPTIONS)
  {
    // This ensures that we test what we intend to test: the Standard requires the strong exception safety
    // guarantee for types that are nothrow move constructible or copy insertable, but that's not what we're
    // trying to test. We're trying to test the stronger libc++ guarantee.
    static_assert(!std::is_nothrow_move_constructible<ThrowingMoveOnly>::value, "");
    static_assert(!std::is_copy_constructible<ThrowingMoveOnly>::value, "");

    std::vector<ThrowingMoveOnly, Allocator<ThrowingMoveOnly> > v;
    v.emplace_back(0, /* do throw */ false);
    v.emplace_back(1, /* do throw */ false);

    while (v.size() < v.capacity()) {
      v.emplace_back(2, /* do throw */ false);
    }
    assert(v.size() == v.capacity()); // the next emplace will be forced to invalidate iterators

    v.emplace(v.cend(), 3, /* do throw */ true); // this shouldn't throw since we shouldn't move this element at all

    assert(v.size() >= 3);
    assert(v[0] == ThrowingMoveOnly(0));
    assert(v[1] == ThrowingMoveOnly(1));
    assert(v.back() == ThrowingMoveOnly(3));
    assert(is_contiguous_container_asan_correct(v));
  }
#endif // defined(_LIBCPP_VERSION) && !defined(TEST_HAS_NO_EXCEPTIONS)
}

TEST_CONSTEXPR_CXX20 bool tests() {
  test<std::allocator, int>();
  test<min_allocator, int>();
  test<safe_allocator, int>();

  test<std::allocator, MoveOnly>();
  test<min_allocator, MoveOnly>();
  test<safe_allocator, MoveOnly>();

  test<std::allocator, NonTriviallyRelocatable>();
  test<min_allocator, NonTriviallyRelocatable>();
  test<safe_allocator, NonTriviallyRelocatable>();

  // test<limited_allocator<int, 7> >();
  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 17
  static_assert(tests());
#endif
  return 0;
}
