//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// class inplace_vector

//   constexpr auto operator<=>(const inplace_vector& x,
//                              const inplace_vector& y);

// FIXME: check if the auto is valid

#include <cassert>
#include <inplace_vector>

#include "test_container_comparisons.h"

template <typename T>
using inplace_vector_size_10 = std::inplace_vector<T, 10>;
template <typename T>
using inplace_vector_size_0 = std::inplace_vector<T, 0>;

// TODO: test inplace_vector<T, 0> still has a valid definition
// TODO: modify so that the checks work in constexpr

int main(int, char**) {
  assert(test_sequence_container_spaceship<inplace_vector_size_10>());
  static_assert(test_sequence_container_spaceship<inplace_vector_size_10>());
  assert(test_sequence_container_spaceship<inplace_vector_size_0>());
  static_assert(test_sequence_container_spaceship<inplace_vector_size_0>());
  return 0;
}
