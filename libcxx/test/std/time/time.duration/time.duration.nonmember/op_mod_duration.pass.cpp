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
//   constexpr
//   typename common_type<duration<Rep1, Period1>, duration<Rep2, Period2>>::type
//   operator%(const duration<Rep1, Period1>& lhs, const duration<Rep2, Period2>& rhs);

#include <chrono>
#include <cassert>
#include <ratio>
#include "../../rep.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    std::chrono::nanoseconds ns1(15);
    std::chrono::nanoseconds ns2(6);
    std::chrono::nanoseconds r = ns1 % ns2;
    assert(r.count() == 3);
    }
    {
    std::chrono::microseconds us1(15);
    std::chrono::nanoseconds ns2(28);
    std::chrono::nanoseconds r = us1 % ns2;
    assert(r.count() == 20);
    }
    {
    std::chrono::duration<int, std::ratio<3, 5> > s1(6);
    std::chrono::duration<int, std::ratio<2, 3> > s2(3);
    std::chrono::duration<int, std::ratio<1, 15> > r = s1 % s2;
    assert(r.count() == 24);
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::chrono::nanoseconds ns1(15);
    constexpr std::chrono::nanoseconds ns2(6);
    constexpr std::chrono::nanoseconds r = ns1 % ns2;
    static_assert(r.count() == 3, "");
    }
    {
    constexpr std::chrono::microseconds us1(15);
    constexpr std::chrono::nanoseconds ns2(28);
    constexpr std::chrono::nanoseconds r = us1 % ns2;
    static_assert(r.count() == 20, "");
    }
    {
    constexpr std::chrono::duration<int, std::ratio<3, 5> > s1(6);
    constexpr std::chrono::duration<int, std::ratio<2, 3> > s2(3);
    constexpr std::chrono::duration<int, std::ratio<1, 15> > r = s1 % s2;
    static_assert(r.count() == 24, "");
    }
    {
      std::chrono::duration<int> d(5);
      RepConstConvertibleLWG3050 x;

      {
        auto r = d % x;
        assert(r.count() == 1);
        ASSERT_SAME_TYPE(std::chrono::duration<long>, decltype(r));
      }
    }
#endif // TEST_STD_VER >= 11

    return 0;
}
