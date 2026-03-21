//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class ForwardIterator, class Size, class T>
//   constexpr ForwardIterator     // constexpr after C++17
//   search_n(ForwardIterator first, ForwardIterator last, Size count,
//            const T& value);

#include <algorithm>
#include <array>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
TEST_CONSTEXPR_CXX20 bool test() {
  { // simple test
    int a[]  = {1, 2, 3, 4, 5, 6};
    auto ret = std::search_n(Iter(a), Iter(a + 6), 1, 3);
    assert(base(ret) == a + 2);
  }
  { // matching part begins at the front
    int a[]  = {7, 7, 3, 7, 3, 6};
    auto ret = std::search_n(Iter(a), Iter(a + 6), 2, 7);
    assert(base(ret) == a);
  }
  { // matching part ends at the back
    int a[]  = {9, 3, 6, 4, 4};
    auto ret = std::search_n(Iter(a), Iter(a + 5), 2, 4);
    assert(base(ret) == a + 3);
  }
  { // pattern does not match
    int a[]  = {9, 3, 6, 4, 8};
    auto ret = std::search_n(Iter(a), Iter(a + 5), 1, 1);
    assert(base(ret) == a + 5);
  }
  { // range and pattern are identical
    int a[]  = {1, 1, 1, 1};
    auto ret = std::search_n(Iter(a), Iter(a + 4), 4, 1);
    assert(base(ret) == a);
  }
  { // pattern is longer than range
    int a[]  = {3, 3, 3};
    auto ret = std::search_n(Iter(a), Iter(a + 3), 4, 3);
    assert(base(ret) == a + 3);
  }
  { // pattern has zero length
    int a[]  = {6, 7, 8};
    auto ret = std::search_n(Iter(a), Iter(a + 3), 0, 7);
    assert(base(ret) == a);
  }
  { // range has zero length
    std::array<int, 0> a = {};
    auto ret             = std::search_n(Iter(a.data()), Iter(a.data()), 1, 1);
    assert(base(ret) == a.data());
  }
  {   // check that the first match is returned
    { // Match is at the start
      int a[]  = {6, 6, 8, 6, 6, 8, 6, 6, 8};
      auto ret = std::search_n(Iter(a), Iter(a + 9), 2, 6);
      assert(base(ret) == a);
    }
    { // Match is in the middle
      int a[]  = {6, 8, 8, 6, 6, 8, 6, 6, 8};
      auto ret = std::search_n(Iter(a), Iter(a + 9), 2, 6);
      assert(base(ret) == a + 3);
    }
    { // Match is at the end
      int a[]  = {6, 6, 8, 6, 6, 8, 6, 6, 6};
      auto ret = std::search_n(Iter(a), Iter(a + 9), 3, 6);
      assert(base(ret) == a + 6);
    }
  }

  return true;
}

int main(int, char**) {
  test<forward_iterator<const int*> >();
  test<bidirectional_iterator<const int*> >();
  test<random_access_iterator<const int*> >();
#if TEST_STD_VER >= 20
  static_assert(test<forward_iterator<const int*> >());
  static_assert(test<bidirectional_iterator<const int*> >());
  static_assert(test<random_access_iterator<const int*> >());
#endif

  return 0;
}
