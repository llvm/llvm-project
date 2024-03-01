//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// inline constexpr size_t hardware_destructive_interference_size = implementation-defined;  // since C++17
// inline constexpr size_t hardware_constructive_interference_size = implementation-defined; // since C++17

// UNSUPPORTED: c++03, c++11, c++14

#include <new>
#include <cassert>

#include "test_macros.h"

constexpr bool test() {
  ASSERT_SAME_TYPE(decltype(std::hardware_destructive_interference_size), const std::size_t);
  ASSERT_SAME_TYPE(decltype(std::hardware_constructive_interference_size), const std::size_t);

#if defined(__APPLE__) && defined(__arm64__)
  assert(std::hardware_destructive_interference_size == 128);
  assert(std::hardware_constructive_interference_size == 128);
#else
  assert(std::hardware_destructive_interference_size == 64);
  assert(std::hardware_constructive_interference_size == 64);
#endif

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
