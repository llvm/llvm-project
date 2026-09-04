//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <deque>

// template<class T, class Allocator>
//   synth-three-way-result<T> operator<=>(const deque<T, Allocator>& x,
//                                         const deque<T, Allocator>& y);

#include <cassert>
#include <deque>

#include "test_container_comparisons.h"

TEST_CONSTEXPR_CXX26 bool test() {
  assert(test_sequence_container_spaceship<std::deque>());
  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER >= 26
  static_assert(test());
#endif
  return 0;
}
