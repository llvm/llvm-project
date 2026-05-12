//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<container-compatible-range<T> R>
//   constexpr inplace_vector(from_range_t, R&& rg);

#include <cassert>
#include <inplace_vector>
#include <ranges>

#include "../common.h"
#include "test_iterators.h"
#include "test_macros.h"

constexpr bool test() {
  {
    int a[] = {1, 2, 3, 4};
    std::inplace_vector<int, 8> c(std::from_range, a);
    assert_inplace_vector_equal(c, a);
  }
  {
    using Iter = cpp20_input_iterator<int*>;
    int a[]    = {1, 2, 3, 4};
    auto r     = std::views::counted(Iter(a), 4);
    std::inplace_vector<int, 8> c(std::from_range, r);
    assert_inplace_vector_equal(c, a);
  }
  {
    int a[] = {1, 2, 3, 4};
    auto r  = std::ranges::subrange(a, a + 4);
    std::inplace_vector<int, 4> c(std::from_range, r);
    assert_inplace_vector_equal(c, a);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
