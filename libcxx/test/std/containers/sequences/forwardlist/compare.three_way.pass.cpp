//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <forward_list>

// template<class T, class Allocator>
//   synth-three-way-result<T> operator<=>(const forward_list<T, Allocator>& x,
//                                         const forward_list<T, Allocator>& y); // constexpr since C++26

#include <cassert>
#include <forward_list>

#include "test_container_comparisons.h"

int main(int, char**) {
  assert(test_sequence_container_spaceship<std::forward_list>());
#if TEST_STD_VER >= 26
  static_assert(test_sequence_container_spaceship<std::forward_list>());
#endif

  return 0;
}
