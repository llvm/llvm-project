//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>

// file_time

#include <chrono>

#include "test_macros.h"

template <class Dur>
void test() {
  ASSERT_SAME_TYPE(std::chrono::file_time<Dur>, std::chrono::time_point<std::chrono::file_clock, Dur>);
}

int main(int, char**) {
  test<std::chrono::nanoseconds>();
  test<std::chrono::minutes>();
  test<std::chrono::hours>();

  return 0;
}
