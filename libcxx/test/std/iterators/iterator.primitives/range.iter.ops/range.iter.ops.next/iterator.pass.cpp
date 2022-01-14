//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// ranges::next(it)

#include <iterator>

#include <cassert>
#include <concepts>
#include <utility>

#include "test_iterators.h"

template <class It>
constexpr void check(int* first, int* expected) {
  It it(first);
  std::same_as<It> auto result = std::ranges::next(std::move(it));
  assert(base(result) == expected);
}

constexpr bool test() {
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int n = 0; n != 9; ++n) {
    check<cpp17_input_iterator<int*>>(  range+n, range+n+1);
    check<cpp20_input_iterator<int*>>(  range+n, range+n+1);
    check<forward_iterator<int*>>(      range+n, range+n+1);
    check<bidirectional_iterator<int*>>(range+n, range+n+1);
    check<random_access_iterator<int*>>(range+n, range+n+1);
    check<contiguous_iterator<int*>>(   range+n, range+n+1);
    check<output_iterator<int*>>(       range+n, range+n+1);
    check<int*>(                        range+n, range+n+1);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
