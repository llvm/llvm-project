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

int main(int, char**) {
  assert(test_sequence_container_spaceship<std::deque>());
  // `std::deque` is not constexpr, so no `static_assert` test here.
  return 0;
}
