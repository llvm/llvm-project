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
// constexpr chrono::seconds seconds() const noexcept;
//
// See the table in hours.pass.cpp for correspondence between the magic values used below

#include <chrono>
#include <cassert>
#include <cstdint>
#include <ratio>
#include <utility>

#include "test_macros.h"

template <typename Duration>
constexpr int64_t check_seconds(Duration d)
{
    using HMS = std::chrono::hh_mm_ss<Duration>;
    ASSERT_SAME_TYPE(std::chrono::seconds, decltype(std::declval<HMS>().seconds()));
    ASSERT_NOEXCEPT(                                std::declval<HMS>().seconds());
    return HMS(d).seconds().count();
}

int main(int, char**)
{
    using microfortnights = std::chrono::duration<int, std::ratio<756, 625>>;

    static_assert( check_seconds(std::chrono::seconds( 1)) == 1, "");
    static_assert( check_seconds(std::chrono::seconds(-1)) == 1, "");

    assert( check_seconds(std::chrono::seconds( 5000)) == 20);
    assert( check_seconds(std::chrono::seconds(-5000)) == 20);
    assert( check_seconds(std::chrono::minutes( 5000)) == 0);
    assert( check_seconds(std::chrono::minutes(-5000)) == 0);
    assert( check_seconds(std::chrono::hours( 11))     == 0);
    assert( check_seconds(std::chrono::hours(-11))     == 0);

    assert( check_seconds(std::chrono::milliseconds( 123456789LL)) == 36);
    assert( check_seconds(std::chrono::milliseconds(-123456789LL)) == 36);
    assert( check_seconds(std::chrono::microseconds( 123456789LL)) == 3);
    assert( check_seconds(std::chrono::microseconds(-123456789LL)) == 3);
    assert( check_seconds(std::chrono::nanoseconds( 123456789LL))  == 0);
    assert( check_seconds(std::chrono::nanoseconds(-123456789LL))  == 0);

    assert( check_seconds(microfortnights(  1000)) ==  9);
    assert( check_seconds(microfortnights( -1000)) ==  9);
    assert( check_seconds(microfortnights( 10000)) == 36);
    assert( check_seconds(microfortnights(-10000)) == 36);

    return 0;
}
