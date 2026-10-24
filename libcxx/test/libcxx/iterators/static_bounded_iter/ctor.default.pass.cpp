//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Iterator, size_t _Size>
// struct __static_bounded_iter;
//
// Default constructor

#include <__iterator/static_bounded_iter.h>
#include <cassert>
#include <cstddef>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter, std::size_t Size>
TEST_CONSTEXPR_CXX14 bool tests() {
  // A default-constructed iterator is a singular iterator. Its members are value-initialized, so
  // comparing two such iterators is well-defined, even when they are default-initialized rather
  // than value-initialized.
  {
    std::__static_bounded_iter<Iter, Size> iter1;
    std::__static_bounded_iter<Iter, Size> iter2;
    assert(iter1 == iter2);
    assert(!(iter1 != iter2));
    assert(!(iter1 < iter2));
    assert(iter1 <= iter2);
    assert(iter1 - iter2 == 0);
  }
  // A default-initialized iterator compares equal to a value-initialized one.
  {
    std::__static_bounded_iter<Iter, Size> default_initialized;
    std::__static_bounded_iter<Iter, Size> value_initialized = std::__static_bounded_iter<Iter, Size>();
    assert(default_initialized == value_initialized);
  }
  // The storage's default constructor is user-provided, so default-initializing a const object is valid.
  {
    std::__static_bounded_iter<Iter, Size> const iter1;
    std::__static_bounded_iter<Iter, Size> const iter2;
    assert(iter1 == iter2);
  }

  return true;
}

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests_all_sizes() {
  tests<Iter, 0>();
  tests<Iter, 1>();
  tests<Iter, 5>();
  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT((std::__static_bounded_iter<int*, 0>()));
  ASSERT_NOEXCEPT((std::__static_bounded_iter<int*, 5>()));

  tests_all_sizes<int*>();
#if TEST_STD_VER > 11
  static_assert(tests_all_sizes<int*>(), "");
#endif

#if TEST_STD_VER > 17
  tests_all_sizes<contiguous_iterator<int*> >();
  static_assert(tests_all_sizes<contiguous_iterator<int*> >(), "");
#endif

  return 0;
}
