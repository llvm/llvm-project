//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <chrono>

// abs

// template <class Rep, class Period>
//   constexpr duration<Rep, Period> abs(duration<Rep, Period> d)

#include <chrono>
#include <cassert>
#include <ratio>
#include <type_traits>

#include "test_macros.h"

template <class Duration>
void
test(const Duration& f, const Duration& d)
{
    {
    typedef decltype(std::chrono::abs(f)) R;
    static_assert((std::is_same<R, Duration>::value), "");
    assert(std::chrono::abs(f) == d);
    }
}

int main(int, char**)
{
//  7290000ms is 2 hours, 1 minute, and 30 seconds
    test(std::chrono::milliseconds( 7290000), std::chrono::milliseconds( 7290000));
    test(std::chrono::milliseconds(-7290000), std::chrono::milliseconds( 7290000));
    test(std::chrono::minutes( 122), std::chrono::minutes( 122));
    test(std::chrono::minutes(-122), std::chrono::minutes( 122));
    test(std::chrono::hours(0), std::chrono::hours(0));

    {
//  9000000ms is 2 hours and 30 minutes
    constexpr std::chrono::hours h1 = std::chrono::abs(std::chrono::hours(-3));
    static_assert(h1.count() == 3, "");
    constexpr std::chrono::hours h2 = std::chrono::abs(std::chrono::hours(3));
    static_assert(h2.count() == 3, "");
    }

    {
//  Make sure it works for durations that are not LCD'ed - example from LWG3091
    constexpr auto d = std::chrono::abs(std::chrono::duration<int, std::ratio<60, 100>>{2});
    static_assert(d.count() == 2, "");
    }

  return 0;
}
