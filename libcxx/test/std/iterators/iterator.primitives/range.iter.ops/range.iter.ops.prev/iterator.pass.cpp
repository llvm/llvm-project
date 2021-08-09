//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::prev(it)

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <class It>
constexpr void check() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  assert(std::ranges::prev(It(&range[1])) == It(&range[0]));
  assert(std::ranges::prev(It(&range[4])) == It(&range[3]));
  assert(std::ranges::prev(It(&range[5])) == It(&range[4]));
  assert(std::ranges::prev(It(&range[6])) == It(&range[5]));
  assert(std::ranges::prev(It(&range[10])) == It(&range[9]));
}

constexpr bool test() {
  check<bidirectional_iterator<int*>>();
  check<random_access_iterator<int*>>();
  check<contiguous_iterator<int*>>();
  check<int*>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
