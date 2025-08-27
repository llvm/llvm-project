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
// Comparison operators

#include <concepts>
#include <__iterator/bounded_iter.h>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
TEST_CONSTEXPR_CXX14 bool tests() {
  int array[]                           = {0, 1, 2, 3, 4};
  int* b                                = array + 0;
  int* e                                = array + 5;
  std::__bounded_iter<Iter> const iter1 = std::__make_bounded_iter(Iter(b), Iter(b), Iter(e));
  std::__bounded_iter<Iter> const iter2 = std::__make_bounded_iter(Iter(e), Iter(b), Iter(e));

  // operator==
  {
    assert(iter1 == iter1);
    assert(!(iter1 == iter2));
  }
  // operator!=
  {
    assert(iter1 != iter2);
    assert(!(iter1 != iter1));
  }
  // operator<
  {
    assert(iter1 < iter2);
    assert(!(iter2 < iter1));
    assert(!(iter1 < iter1));
  }
  // operator>
  {
    assert(iter2 > iter1);
    assert(!(iter1 > iter2));
    assert(!(iter1 > iter1));
  }
  // operator<=
  {
    assert(iter1 <= iter2);
    assert(!(iter2 <= iter1));
    assert(iter1 <= iter1);
  }
  // operator>=
  {
    assert(iter2 >= iter1);
    assert(!(iter1 >= iter2));
    assert(iter1 >= iter1);
  }

#if TEST_STD_VER >= 20
  // P1614
  std::same_as<std::strong_ordering> decltype(auto) r1 = iter1 <=> iter2;
  assert(r1 == std::strong_ordering::less);
#endif

  return true;
}

int main(int, char**) {
  tests<int*>();
#if TEST_STD_VER > 11
  static_assert(tests<int*>(), "");
#endif

#if TEST_STD_VER > 17
  tests<contiguous_iterator<int*>>();
  static_assert(tests<contiguous_iterator<int*>>());

  tests<three_way_contiguous_iterator<int*>>();
  static_assert(tests<three_way_contiguous_iterator<int*>>());
#endif

  return 0;
}
