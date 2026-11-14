//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Default constructor

#include <__iterator/bounded_iter.h>
#include <cassert>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  // A default-constructed iterator is a singular iterator pointing past the end of an empty range.
  // Its members are value-initialized, so comparing two such iterators is well-defined, even when
  // they are default-initialized rather than value-initialized.
  {
    std::__bounded_iter<Iter> iter1;
    std::__bounded_iter<Iter> iter2;
    assert(iter1 == iter2);
    assert(!(iter1 != iter2));
    assert(!(iter1 < iter2));
    assert(iter1 <= iter2);
    assert(iter1 - iter2 == 0);
  }
  // A default-initialized iterator compares equal to a value-initialized one.
  {
    std::__bounded_iter<Iter> default_initialized;
    std::__bounded_iter<Iter> value_initialized = std::__bounded_iter<Iter>();
    assert(default_initialized == value_initialized);
  }
  // The default constructor is user-provided, so default-initializing a const object is valid.
  {
    std::__bounded_iter<Iter> const iter1;
    std::__bounded_iter<Iter> const iter2;
    assert(iter1 == iter2);
  }

  return true;
}

int main(int, char**) {
  ASSERT_NOEXCEPT(std::__bounded_iter<int*>());

  tests<int*>();
#if TEST_STD_VER > 11
  static_assert(tests<int*>(), "");
#endif

#if TEST_STD_VER > 17
  tests<contiguous_iterator<int*> >();
  static_assert(tests<contiguous_iterator<int*> >(), "");
#endif

  return 0;
}
