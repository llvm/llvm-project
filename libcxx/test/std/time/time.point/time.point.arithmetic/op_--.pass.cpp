//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// REQUIRES: std-at-least-c++20

// <chrono>

// time_point

// constexpr time_point& operator--();

#include <cassert>
#include <chrono>

#include "test_macros.h"

constexpr bool test() {
  using Clock    = std::chrono::system_clock;
  using Duration = std::chrono::milliseconds;
  std::chrono::time_point<Clock, Duration> t{Duration{5}};
  std::chrono::time_point<Clock, Duration>& tref{--t};
  assert(&tref == &t);
  assert(tref.time_since_epoch() == Duration{4});
  return true;
}

int main(int, char**) {
  test();
  static_assert(test());
  return 0;
}
