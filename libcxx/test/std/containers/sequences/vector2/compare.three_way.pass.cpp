//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++29

// <vector>

// class vector2

// template<class T, class Allocator>
//   constexpr synth-three-way-result<T> operator<=>(const vector2<T, Allocator>& x,
//                                                   const vector2<T, Allocator>& y);

#include <cassert>
#include <vector>

#include "test_container_comparisons.h"

int main(int, char**) {
  assert(test_sequence_container_spaceship<std::vector2>());
  static_assert(test_sequence_container_spaceship<std::vector2>());
  return 0;
}
