//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period1, class Rep2, class Period2>
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator+(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    std::chrono::seconds s1(3);
    std::chrono::seconds s2(5);
    std::chrono::seconds r = s1 + s2;
    assert(r.count() == 8);
    }
    {
    std::chrono::seconds s1(3);
    std::chrono::microseconds s2(5);
    std::chrono::microseconds r = s1 + s2;
    assert(r.count() == 3000005);
    }
    {
    std::chrono::duration<int, std::ratio<2, 3> > s1(3);
    std::chrono::duration<int, std::ratio<3, 5> > s2(5);
    std::chrono::duration<int, std::ratio<1, 15> > r = s1 + s2;
    assert(r.count() == 75);
    }
    {
    std::chrono::duration<int, std::ratio<2, 3> > s1(3);
    std::chrono::duration<double, std::ratio<3, 5> > s2(5);
    std::chrono::duration<double, std::ratio<1, 15> > r = s1 + s2;
    assert(r.count() == 75);
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::chrono::seconds s1(3);
    constexpr std::chrono::seconds s2(5);
    constexpr std::chrono::seconds r = s1 + s2;
    static_assert(r.count() == 8, "");
    }
    {
    constexpr std::chrono::seconds s1(3);
    constexpr std::chrono::microseconds s2(5);
    constexpr std::chrono::microseconds r = s1 + s2;
    static_assert(r.count() == 3000005, "");
    }
    {
    constexpr std::chrono::duration<int, std::ratio<2, 3> > s1(3);
    constexpr std::chrono::duration<int, std::ratio<3, 5> > s2(5);
    constexpr std::chrono::duration<int, std::ratio<1, 15> > r = s1 + s2;
    static_assert(r.count() == 75, "");
    }
    {
    constexpr std::chrono::duration<int, std::ratio<2, 3> > s1(3);
    constexpr std::chrono::duration<double, std::ratio<3, 5> > s2(5);
    constexpr std::chrono::duration<double, std::ratio<1, 15> > r = s1 + s2;
    static_assert(r.count() == 75, "");
    }
#endif

  return 0;
}
