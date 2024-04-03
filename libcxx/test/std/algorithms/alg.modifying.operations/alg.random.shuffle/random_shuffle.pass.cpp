//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter>
//   void
//   random_shuffle(Iter first, Iter last);

// REQUIRES: c++03 || c++11 || c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <algorithm>
#include <array>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
void
test_with_iterator()
{
  std::array<int, 0> empty = {};
  std::random_shuffle(Iter(empty.data()), Iter(empty.data()));

  const int all_elements[] = {1, 2, 3, 4};
  int shuffled[]           = {1, 2, 3, 4};
  const unsigned size      = sizeof(all_elements) / sizeof(all_elements[0]);

  std::random_shuffle(Iter(shuffled), Iter(shuffled + size));
  assert(std::is_permutation(shuffled, shuffled + size, all_elements));

  std::random_shuffle(Iter(shuffled), Iter(shuffled + size));
  assert(std::is_permutation(shuffled, shuffled + size, all_elements));
}


int main(int, char**)
{
    int ia[]  = {1, 2, 3, 4};
    int ia1[] = {1, 4, 3, 2};
    int ia2[] = {4, 1, 2, 3};
    const unsigned sa = sizeof(ia)/sizeof(ia[0]);

    std::random_shuffle(ia, ia+sa);
    LIBCPP_ASSERT(std::equal(ia, ia+sa, ia1));
    assert(std::is_permutation(ia, ia+sa, ia1));

    std::random_shuffle(ia, ia+sa);
    LIBCPP_ASSERT(std::equal(ia, ia+sa, ia2));
    assert(std::is_permutation(ia, ia+sa, ia2));

    test_with_iterator<random_access_iterator<int*> >();
    test_with_iterator<int*>();

  return 0;
}
