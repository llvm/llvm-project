//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts

// Iterator conformance tests for counted_iterator.

#include <iterator>

#include "test_macros.h"
#include "test_iterators.h"

void test() {
  static_assert(std::input_iterator<std::counted_iterator<cpp17_input_iterator<int*>>>);
  static_assert(std::forward_iterator<std::counted_iterator<forward_iterator<int*>>>);
  static_assert(std::bidirectional_iterator<std::counted_iterator<random_access_iterator<int*>>>);
  static_assert(std::bidirectional_iterator<std::counted_iterator<contiguous_iterator<int*>>>);
  static_assert(std::random_access_iterator<std::counted_iterator<random_access_iterator<int*>>>);
  static_assert(std::contiguous_iterator<std::counted_iterator<contiguous_iterator<int*>>>);

  using Iter = std::counted_iterator<forward_iterator<int*>>;
  static_assert(std::indirectly_writable<Iter, int>);
  static_assert(std::indirectly_swappable<Iter, Iter>);
}
