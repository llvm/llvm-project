//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26

// <inplace_vector>

// template<class T, size_t N, class Predicate>
//   constexpr typename inplace_vector<T, N>::size_type
//     erase_if(inplace_vector<T, N>& c, Predicate pred);

#include <cassert>
#include <inplace_vector>

#include "../common.h"
#include "test_macros.h"

constexpr bool test() {
  std::inplace_vector<int, 8> c{1, 2, 3, 4, 5, 6};
  ASSERT_SAME_TYPE(
      std::inplace_vector<int, 8>::size_type, decltype(std::erase_if(c, [](int v) { return v % 2 == 0; })));
  auto erased = std::erase_if(c, [](int v) { return v % 2 == 0; });
  assert(erased == 3);
  assert_inplace_vector_equal(c, {1, 3, 5});

  erased = std::erase_if(c, [](int v) { return v > 10; });
  assert(erased == 0);
  assert_inplace_vector_equal(c, {1, 3, 5});

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
