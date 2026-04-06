//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class RandomAccessIterator, class RandomFunc>
//   void random_shuffle(RandomAccessIterator first, RandomAccessIterator last, RandomFunc& rand); // until C++11
//
// template<class RandomAccessIterator, class RandomFunc>
//   void random_shuffle(RandomAccessIterator first, RandomAccessIterator last, RandomFunc&& rand); // since C++11

// REQUIRES: c++03 || c++11 || c++14
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

struct RandomGenerator {
  std::ptrdiff_t operator()(std::ptrdiff_t n) { return n - 1; }
};

template <class Iter>
void test_with_iterator() {
  RandomGenerator gen;
  int values[] = {1, 2, 3, 4};

  // Make sure the algorithm shuffles
  {
    int arr[] = {1, 2, 3, 4};
    std::random_shuffle(Iter(std::begin(arr)), Iter(std::end(arr)), gen);
    int libcxx_expected[] = {4, 1, 2, 3};
    LIBCPP_ASSERT(std::equal(std::begin(arr), std::end(arr), std::begin(libcxx_expected)));
    assert(std::is_permutation(std::begin(arr), std::end(arr), std::begin(values)));
    std::random_shuffle(Iter(std::begin(arr)), Iter(std::end(arr)), gen);
    assert(std::is_permutation(std::begin(arr), std::end(arr), std::begin(values)));
  }

  // Test the signature in C++03 mode, which takes a lvalue reference
#if TEST_STD_VER == 03
  {
    int arr[] = {1, 2, 3, 4};
    std::random_shuffle<Iter, RandomGenerator>(Iter(std::begin(arr)), Iter(std::end(arr)), gen);
    assert(std::is_permutation(std::begin(arr), std::end(arr), std::begin(values)));
  }
#endif
}

int main(int, char**) {
  test_with_iterator<random_access_iterator<int*> >();
  test_with_iterator<int*>();
  return 0;
}
