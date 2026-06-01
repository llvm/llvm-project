//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <deque>

// constexpr since C++26

// template<class T, class Allocator>
//   synth-three-way-result<T> operator<=>(const deque<T, Allocator>& x,
//                                         const deque<T, Allocator>& y);

#include <cassert>
#include <deque>

#include "test_container_comparisons.h"
#include "test_macros.h"

#if TEST_STD_VER >= 26
TEST_CONSTEXPR_CXX26 bool test_constexpr() {
  std::deque<int> a = {1, 2, 3};
  std::deque<int> b = {1, 2, 4};
  assert((a <=> b) < 0);
  return true;
}
#endif

int main(int, char**) {
#if TEST_STD_VER >= 26
  assert(test_constexpr());
  static_assert(test_constexpr());
#endif

  assert(test_sequence_container_spaceship<std::deque>());
  return 0;
}
