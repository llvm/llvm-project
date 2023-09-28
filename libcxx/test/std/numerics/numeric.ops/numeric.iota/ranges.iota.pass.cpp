//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Testing std::ranges::iota

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include <cassert>
#include <numeric>
#include <array>

#include "test_macros.h"
#include "test_iterators.h"

// This is pulled directly from the std::iota test
template <class InIter>
TEST_CONSTEXPR_CXX20 void test0() {
  int ia[]         = {1, 2, 3, 4, 5};
  int ir[]         = {5, 6, 7, 8, 9};
  const unsigned s = sizeof(ia) / sizeof(ia[0]);
  std::ranges::iota(InIter(ia), InIter(ia + s), 5);
  for (unsigned i = 0; i < s; ++i)
    assert(ia[i] == ir[i]);
}

TEST_CONSTEXPR_CXX20 bool test0() {
  test0<forward_iterator<int*> >();
  test0<bidirectional_iterator<int*> >();
  test0<random_access_iterator<int*> >();
  test0<int*>();

  return true;
}

TEST_CONSTEXPR_CXX20 void test1() {
  std::array<int, 5> ia = {1, 2, 3, 4, 5};
  std::array<int, 5> ir = {5, 6, 7, 8, 9};
  std::ranges::iota(ia, 5);
  for (unsigned i = 0; i < ir.size(); ++i)
    assert(ia[i] == ir[i]);
}

int main(int, char**) {
  test0();
  test1();
  return 0;
}