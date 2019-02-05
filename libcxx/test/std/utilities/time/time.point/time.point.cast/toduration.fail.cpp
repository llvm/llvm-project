//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// template <class ToDuration, class Clock, class Duration>
//   time_point<Clock, ToDuration>
//   time_point_cast(const time_point<Clock, Duration>& t);

// ToDuration shall be an instantiation of duration.

#include <chrono>

int main(int, char**)
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::time_point<Clock, std::chrono::milliseconds> FromTimePoint;
    typedef std::chrono::time_point<Clock, std::chrono::minutes> ToTimePoint;
    std::chrono::time_point_cast<ToTimePoint>(FromTimePoint(std::chrono::milliseconds(3)));

  return 0;
}
