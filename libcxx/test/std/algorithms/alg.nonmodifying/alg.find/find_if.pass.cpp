//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<InputIterator Iter, Predicate<auto, Iter::value_type> Pred>
//   requires CopyConstructible<Pred>
//   constexpr Iter   // constexpr after C++17
//   find_if(Iter first, Iter last, Pred pred);

#include <algorithm>
#include <functional>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

struct EqualTo {
  int v;
  TEST_CONSTEXPR EqualTo(int val) : v(val) {}
  TEST_CONSTEXPR bool operator()(int other) const { return v == other; }
};

template <class Iter>
TEST_CONSTEXPR_CXX17 void test_iter() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  // We don't find what we're looking for in the range
  {
    {
      Iter result = std::find_if(Iter(range), Iter(range), EqualTo(0));
      assert(result == Iter(range));
    }
    {
      Iter result = std::find_if(Iter(range), Iter(std::end(range)), EqualTo(999));
      assert(result == Iter(std::end(range)));
    }
  }

  // Tests with sub-ranges of various sizes
  for (int size = 1; size != 10; ++size) {
    for (int i = 0; i != size - 1; ++i) {
      Iter result = std::find_if(Iter(range), Iter(range + size), EqualTo(i));
      assert(result == Iter(range + i));
    }
  }
}

TEST_CONSTEXPR_CXX17 bool test() {
  test_iter<cpp17_input_iterator<int*> >();
  test_iter<forward_iterator<int*> >();
  test_iter<bidirectional_iterator<int*> >();
  test_iter<random_access_iterator<int*> >();
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 20
  static_assert(test());
#endif

  return 0;
}
