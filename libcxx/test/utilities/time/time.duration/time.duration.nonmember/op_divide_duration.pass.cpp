//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   typename common_type<Rep1, Rep2>::type
//   operator/(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <chrono>
#include <cassert>

int main()
{
    {
    std::chrono::nanoseconds ns1(15);
    std::chrono::nanoseconds ns2(5);
    assert(ns1 / ns2 == 3);
    }
    {
    std::chrono::microseconds us1(15);
    std::chrono::nanoseconds ns2(5);
    assert(us1 / ns2 == 3000);
    }
    {
    std::chrono::duration<int, std::ratio<2, 3> > s1(30);
    std::chrono::duration<int, std::ratio<3, 5> > s2(5);
    assert(s1 / s2 == 6);
    }
    {
    std::chrono::duration<int, std::ratio<2, 3> > s1(30);
    std::chrono::duration<double, std::ratio<3, 5> > s2(5);
    assert(s1 / s2 == 20./3);
    }
}
