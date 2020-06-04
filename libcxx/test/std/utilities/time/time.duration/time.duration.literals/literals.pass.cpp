//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <chrono>

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using namespace std::literals::chrono_literals;

//    Make sure the types are right
    static_assert ( std::is_same<decltype( 3h   ), std::chrono::hours>::value, "" );
    static_assert ( std::is_same<decltype( 3min ), std::chrono::minutes>::value, "" );
    static_assert ( std::is_same<decltype( 3s   ), std::chrono::seconds>::value, "" );
    static_assert ( std::is_same<decltype( 3ms  ), std::chrono::milliseconds>::value, "" );
    static_assert ( std::is_same<decltype( 3us  ), std::chrono::microseconds>::value, "" );
    static_assert ( std::is_same<decltype( 3ns  ), std::chrono::nanoseconds>::value, "" );

    std::chrono::hours h = 4h;
    assert ( h == std::chrono::hours(4));
    auto h2 = 4.0h;
    assert ( h == h2 );

    std::chrono::minutes min = 36min;
    assert ( min == std::chrono::minutes(36));
    auto min2 = 36.0min;
    assert ( min == min2 );

    std::chrono::seconds s = 24s;
    assert ( s == std::chrono::seconds(24));
    auto s2 = 24.0s;
    assert ( s == s2 );

    std::chrono::milliseconds ms = 247ms;
    assert ( ms == std::chrono::milliseconds(247));
    auto ms2 = 247.0ms;
    assert ( ms == ms2 );

    std::chrono::microseconds us = 867us;
    assert ( us == std::chrono::microseconds(867));
    auto us2 = 867.0us;
    assert ( us == us2 );

    std::chrono::nanoseconds ns = 645ns;
    assert ( ns == std::chrono::nanoseconds(645));
    auto ns2 = 645.ns;
    assert ( ns == ns2 );


  return 0;
}
