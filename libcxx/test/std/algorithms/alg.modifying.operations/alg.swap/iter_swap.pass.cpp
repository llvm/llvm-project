//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<Iterator Iter1, Iterator Iter2>
//   requires HasSwap<Iter1::reference, Iter2::reference>
//   void
//   iter_swap(Iter1 a, Iter2 b);

#include <algorithm>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "type_algorithms.h"

template <class Iter1>
struct Test {
  template <class Iter2>
  TEST_CONSTEXPR_CXX20 void operator()() {
    int i = 1;
    int j = 2;
    std::iter_swap(Iter1(&i), Iter2(&j));
    assert(i == 2 && j == 1);
  }
};

struct TestIterators {
  template <class Iter>
  TEST_CONSTEXPR_CXX20 void operator()() {
    types::for_each(types::forward_iterator_list<int*>(), Test<Iter>());
  }
};

TEST_CONSTEXPR_CXX20 bool test() {
  types::for_each(types::forward_iterator_list<int*>(), TestIterators());
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
