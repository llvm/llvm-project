//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// template <class Clock, class Duration1, class Duration2>
//   typename common_type<Duration1, Duration2>::type
//   operator-(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <chrono>
#include <cassert>

int main()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds Duration1;
    typedef std::chrono::microseconds Duration2;
    std::chrono::time_point<Clock, Duration1> t1(Duration1(3));
    std::chrono::time_point<Clock, Duration2> t2(Duration2(5));
    assert((t1 - t2) == Duration2(2995));
}
