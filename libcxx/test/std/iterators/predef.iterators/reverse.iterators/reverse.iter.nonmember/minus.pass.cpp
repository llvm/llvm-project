//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iterator>

// reverse_iterator

// template <RandomAccessIterator Iter1, RandomAccessIterator Iter2>
//   requires HasMinus<Iter2, Iter1>
// auto operator-(const reverse_iterator<Iter1>& x, const reverse_iterator<Iter2>& y) // constexpr in C++17
//  -> decltype(y.base() - x.base());

#include <iterator>
#include <cstddef>
#include <cassert>
#include <type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class, class, class = void>
struct HasMinus : std::false_type {};
template <class R1, class R2>
struct HasMinus<R1, R2, decltype((R1() - R2(), void()))> : std::true_type {};

// Test non-subtractable base iterator types
static_assert(HasMinus<std::reverse_iterator<int*>, std::reverse_iterator<int*> >::value, "");
static_assert(HasMinus<std::reverse_iterator<int*>, std::reverse_iterator<const int*> >::value, "");

#if TEST_STD_VER >= 11
static_assert(!HasMinus<std::reverse_iterator<int*>, std::reverse_iterator<char*> >::value, "");
static_assert(!HasMinus<std::reverse_iterator<bidirectional_iterator<int*> >,
                        std::reverse_iterator<bidirectional_iterator<int*> > >::value,
              "");
#endif

template <class It1, class It2>
TEST_CONSTEXPR_CXX17 void test_one(It1 l, It2 r, std::ptrdiff_t x) {
  const std::reverse_iterator<It1> r1(l);
  const std::reverse_iterator<It2> r2(r);
  assert((r1 - r2) == x);
}

template <class Iter>
TEST_CONSTEXPR_CXX17 void test() {
  // Test same base iterator type
  char s[3] = {0};

  test_one(Iter(s), Iter(s), 0);
  test_one(Iter(s), Iter(s + 1), 1);
  test_one(Iter(s + 1), Iter(s), -1);
}

TEST_CONSTEXPR_CXX17 bool tests() {
  {
    test<char*>();
    test<random_access_iterator<char*> >();
#if TEST_STD_VER >= 20
    test<cpp20_random_access_iterator<char*>>();
#endif
  }
  {
    // Test different (but subtractable) base iterator types
    using PC  = const char*;
    char s[3] = {0};
    test_one(PC(s), s, 0);
    test_one(PC(s), s + 1, 1);
    test_one(PC(s + 1), s, -1);
  }

  return true;
}

int main(int, char**) {
  tests();
#if TEST_STD_VER > 14
  static_assert(tests(), "");
#endif
  return 0;
}
