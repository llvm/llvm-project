//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// template <class Duration2>
//   time_point(const time_point<clock, Duration2>& t);

// Duration2 shall be implicitly convertible to duration.

#include <chrono>

int main()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds Duration1;
    typedef std::chrono::microseconds Duration2;
    {
    std::chrono::time_point<Clock, Duration2> t2(Duration2(3));
    std::chrono::time_point<Clock, Duration1> t1 = t2;
    }
}
