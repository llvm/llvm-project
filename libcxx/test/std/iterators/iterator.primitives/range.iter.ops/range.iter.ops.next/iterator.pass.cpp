//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: gcc-10

// ranges::next(it)

#include <iterator>
#include <cassert>

#include "test_iterators.h"

template <class It>
constexpr void check() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  assert(&*std::ranges::next(It(&range[0])) == &range[1]);
  assert(&*std::ranges::next(It(&range[1])) == &range[2]);
  assert(&*std::ranges::next(It(&range[2])) == &range[3]);
  assert(&*std::ranges::next(It(&range[3])) == &range[4]);
  assert(&*std::ranges::next(It(&range[4])) == &range[5]);
  assert(&*std::ranges::next(It(&range[5])) == &range[6]);
}

constexpr bool test() {
  check<cpp17_input_iterator<int*>>();
  check<cpp20_input_iterator<int*>>();
  check<forward_iterator<int*>>();
  check<bidirectional_iterator<int*>>();
  check<random_access_iterator<int*>>();
  check<contiguous_iterator<int*>>();
  check<output_iterator<int*>>();
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
