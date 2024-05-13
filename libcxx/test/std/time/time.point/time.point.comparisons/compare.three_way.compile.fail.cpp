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

// template<class Clock, class Duration1,
//          three_way_comparable_with<Duration1> Duration2>
//   constexpr auto operator<=>(const time_point<Clock, Duration1>& lhs,
//                              const time_point<Clock, Duration2>& rhs);

// time_points with different clocks should not compare

#include <chrono>

#include "../../clock.h"

int main(int, char**) {
  using namespace std::chrono_literals;
  std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> t1{3ms};
  std::chrono::time_point<Clock, std::chrono::milliseconds> t2{3ms};

  t1 <=> t2;

  return 0;
}
