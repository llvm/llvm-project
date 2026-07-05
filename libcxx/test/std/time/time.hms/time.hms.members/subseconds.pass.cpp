//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// <chrono>

// template <class Duration>
// class hh_mm_ss
//
// constexpr precision subseconds() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <chrono>
#include <cassert>
#include <cstdint>
#include <ratio>
#include <utility>

#include "test_macros.h"

template <typename Duration>
constexpr std::int64_t check_subseconds(Duration d)
{
    using HMS = std::chrono::hh_mm_ss<Duration>;
    ASSERT_SAME_TYPE(typename HMS::precision, decltype(std::declval<HMS>().subseconds()));
    ASSERT_NOEXCEPT(                                   std::declval<HMS>().subseconds());
    return HMS(d).subseconds().count();
}

int main(int, char**)
{
    using microfortnights = std::chrono::duration<int, std::ratio<756, 625>>;

    static_assert( check_subseconds(std::chrono::seconds( 1)) == 0, "");
    static_assert( check_subseconds(std::chrono::seconds(-1)) == 0, "");

    assert( check_subseconds(std::chrono::seconds( 5000)) == 0);
    assert( check_subseconds(std::chrono::seconds(-5000)) == 0);
    assert( check_subseconds(std::chrono::minutes( 5000)) == 0);
    assert( check_subseconds(std::chrono::minutes(-5000)) == 0);
    assert( check_subseconds(std::chrono::hours( 11))     == 0);
    assert( check_subseconds(std::chrono::hours(-11))     == 0);

    assert( check_subseconds(std::chrono::milliseconds( 123456789LL)) == 789);
    assert( check_subseconds(std::chrono::milliseconds(-123456789LL)) == 789);
    assert( check_subseconds(std::chrono::microseconds( 123456789LL)) == 456789LL);
    assert( check_subseconds(std::chrono::microseconds(-123456789LL)) == 456789LL);
    assert( check_subseconds(std::chrono::nanoseconds( 123456789LL))  == 123456789LL);
    assert( check_subseconds(std::chrono::nanoseconds(-123456789LL))  == 123456789LL);

    assert( check_subseconds(microfortnights(  1000)) == 6000);
    assert( check_subseconds(microfortnights( -1000)) == 6000);
    assert( check_subseconds(microfortnights( 10000)) == 0);
    assert( check_subseconds(microfortnights(-10000)) == 0);

    return 0;
}
