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

// constexpr counted_iterator() requires default_initializable<I> = default;

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

constexpr bool test() {
  static_assert( std::default_initializable<std::counted_iterator<cpp17_input_iterator<int*>>>);
  static_assert(!std::default_initializable<std::counted_iterator<cpp20_input_iterator<int*>>>);

  std::counted_iterator<cpp17_input_iterator<int*>> iter;
  assert(iter.base() == cpp17_input_iterator<int*>());
  assert(iter.count() == 0);

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
