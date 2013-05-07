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
//   bool
//   operator< (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator> (const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator<=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

// template <class Clock, class Duration1, class Duration2>
//   bool
//   operator>=(const time_point<Clock, Duration1>& lhs, const time_point<Clock, Duration2>& rhs);

#include <chrono>
#include <cassert>

int main()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds Duration1;
    typedef std::chrono::microseconds Duration2;
    typedef std::chrono::time_point<Clock, Duration1> T1;
    typedef std::chrono::time_point<Clock, Duration2> T2;

    {
    T1 t1(Duration1(3));
    T1 t2(Duration1(3));
    assert(!(t1 <  t2));
    assert(!(t1 >  t2));
    assert( (t1 <= t2));
    assert( (t1 >= t2));
    }
    {
    T1 t1(Duration1(3));
    T1 t2(Duration1(4));
    assert( (t1 <  t2));
    assert(!(t1 >  t2));
    assert( (t1 <= t2));
    assert(!(t1 >= t2));
    }
    {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3000));
    assert(!(t1 <  t2));
    assert(!(t1 >  t2));
    assert( (t1 <= t2));
    assert( (t1 >= t2));
    }
    {
    T1 t1(Duration1(3));
    T2 t2(Duration2(3001));
    assert( (t1 <  t2));
    assert(!(t1 >  t2));
    assert( (t1 <= t2));
    assert(!(t1 >= t2));
    }
}
