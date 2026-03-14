//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <vector>

// class vector

// template<class T, class Allocator>
//   constexpr synth-three-way-result<T> operator<=>(const vector<T, Allocator>& x,
//                                                   const vector<T, Allocator>& y);

#include <cassert>
#include <vector>

#include "test_container_comparisons.h"

int main(int, char**) {
  assert(test_sequence_container_spaceship<std::vector>());
  static_assert(test_sequence_container_spaceship<std::vector>());
  return 0;
}
