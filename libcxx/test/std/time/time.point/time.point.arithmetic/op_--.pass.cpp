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
  typedef std::chrono::system_clock Clock;
  typedef std::chrono::milliseconds Duration;
  std::chrono::time_point<Clock, Duration> t{Duration{5}};
  std::chrono::time_point<Clock, Duration>& tref{--t};
  return &tref == &t && tref.time_since_epoch() == Duration{4};
}

int main(int, char**) {
  assert(test());
  static_assert(test());
  return 0;
}
