//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   typename common_type<Duration1, Duration2>::type
//   operator-(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds Duration1;
    typedef std::chrono::microseconds Duration2;
    {
    std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    std::chrono::time_point<Clock, Duration2> t2(Duration2(5));
    assert((t1 - t2) == Duration2(2995));
    }
#if TEST_STD_VER > 11
    {
    constexpr std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    constexpr std::chrono::time_point<Clock, Duration2> t2(Duration2(5));
    static_assert((t1 - t2) == Duration2(2995), "");
    }
#endif
}
