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
// std::pointer_traits specialization

#include <__iterator/bounded_iter.h>
#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  using BoundedIter = std::__bounded_iter<Iter>;
  using PointerTraits = std::pointer_traits<BoundedIter>;
  using BasePointerTraits = std::pointer_traits<Iter>;
  static_assert(std::is_same<typename PointerTraits::pointer, BoundedIter>::value, "");
  static_assert(std::is_same<typename PointerTraits::element_type, typename BasePointerTraits::element_type>::value, "");
  static_assert(std::is_same<typename PointerTraits::difference_type, typename BasePointerTraits::difference_type>::value, "");

  {
    int array[]                           = {0, 1, 2, 3, 4};
    int* b                                = array + 0;
    int* e                                = array + 5;
    std::__bounded_iter<Iter> const iter1 = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
    std::__bounded_iter<Iter> const iter2 = std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));
    assert(std::__to_address(iter1) == b); // in-bounds iterator
    assert(std::__to_address(iter2) == e); // out-of-bounds iterator
#if TEST_STD_VER > 17
    assert(std::to_address(iter1) == b); // in-bounds iterator
    assert(std::to_address(iter2) == e); // out-of-bounds iterator
#endif
  }

  return true;
}

int main(int, char**) {
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
