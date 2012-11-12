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

// template <class Rep2, class Period2>
//   duration(const duration<Rep2, Period2>& d);

// exact conversions allowed for integral reps

#include <chrono>
#include <cassert>

int main()
{
    {
    std::chrono::milliseconds ms(1);
    std::chrono::microseconds us = ms;
    assert(us.count() == 1000);
    }
#ifndef _LIBCPP_HAS_NO_CONSTEXPR
    {
    constexpr std::chrono::milliseconds ms(1);
    constexpr std::chrono::microseconds us = ms;
    static_assert(us.count() == 1000, "");
    }
#endif
}
