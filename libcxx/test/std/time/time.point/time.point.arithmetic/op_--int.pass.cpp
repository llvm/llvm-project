//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// time_point

// constexpr time_point operator--(int);

#include <chrono>
#include <cassert>

#include "test_macros.h"

constexpr bool test_constexpr() {
  typedef std::chrono::system_clock Clock;
  typedef std::chrono::milliseconds Duration;
  std::chrono::time_point<Clock, Duration> t1(Duration(3));
  std::chrono::time_point<Clock, Duration> t2 = t1--;
  return t1.time_since_epoch() == Duration(2) && t2.time_since_epoch() == Duration(3);
}

int main(int, char**) {
  typedef std::chrono::system_clock Clock;
  typedef std::chrono::milliseconds Duration;
  std::chrono::time_point<Clock, Duration> t1(Duration(3));
  std::chrono::time_point<Clock, Duration> t2 = t1--;
  assert(t1.time_since_epoch() == Duration(2));
  assert(t2.time_since_epoch() == Duration(3));

  static_assert(test_constexpr());

  return 0;
}
