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

// constexpr time_point& operator++();

#include <chrono>
#include <cassert>

#include "test_macros.h"

constexpr bool constexpr_test() {
  typedef std::chrono::system_clock Clock;
  typedef std::chrono::milliseconds Duration;
  std::chrono::time_point<Clock, Duration> t(Duration(5));
  return (++t).time_since_epoch() == Duration(6);
}

int main(int, char**) {
  typedef std::chrono::system_clock Clock;
  typedef std::chrono::milliseconds Duration;
  std::chrono::time_point<Clock, Duration> t(Duration(3));
  std::chrono::time_point<Clock, Duration>& tref = ++t;
  assert(&tref == &t);
  assert(t.time_since_epoch() == Duration(4));

  static_assert(constexpr_test());

  return 0;
}
