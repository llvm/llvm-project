//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <chrono>

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using namespace std::literals::chrono_literals;

//    Make sure the types are right
    static_assert ( std::is_same<decltype( 3h   ), std::chrono::hours>::value, "" );
    static_assert ( std::is_same<decltype( 3min ), std::chrono::minutes>::value, "" );
    static_assert ( std::is_same<decltype( 3s   ), std::chrono::seconds>::value, "" );
    static_assert ( std::is_same<decltype( 3ms  ), std::chrono::milliseconds>::value, "" );
    static_assert ( std::is_same<decltype( 3us  ), std::chrono::microseconds>::value, "" );
    static_assert ( std::is_same<decltype( 3ns  ), std::chrono::nanoseconds>::value, "" );
#if TEST_STD_VER > 17
//     static_assert ( std::is_same<decltype( Sunday ),    std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Monday ),    std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Tuesday ),   std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Wednesday ), std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Thursday ),  std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Friday ),    std::chrono::weekday>::value, "" );
//     static_assert ( std::is_same<decltype( Saturday ),  std::chrono::weekday>::value, "" );
// 
//     static_assert ( std::is_same<decltype( January ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( February ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( March ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( April ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( May ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( June ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( July ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( August ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( September ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( October ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( November ), std::chrono::month>::value, "" );
//     static_assert ( std::is_same<decltype( December ), std::chrono::month>::value, "" );

//     static_assert ( std::is_same<decltype( 4d    ), std::chrono::days>::value, "" );
//     static_assert ( std::is_same<decltype( 2016y ), std::chrono::years>::value, "" );
#endif

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


#if TEST_STD_VER > 17
//  assert(std::chrono::Sunday    == std::chrono::weekday(0));
//  assert(std::chrono::Monday    == std::chrono::weekday(1));
//  assert(std::chrono::Tuesday   == std::chrono::weekday(2));
//  assert(std::chrono::Wednesday == std::chrono::weekday(3));
//  assert(std::chrono::Thursday  == std::chrono::weekday(4));
//  assert(std::chrono::Friday    == std::chrono::weekday(5));
//  assert(std::chrono::Saturday  == std::chrono::weekday(6));

	assert(std::chrono::January   == std::chrono::month(1));
	assert(std::chrono::February  == std::chrono::month(2));
	assert(std::chrono::March     == std::chrono::month(3));
	assert(std::chrono::April     == std::chrono::month(4));
	assert(std::chrono::May       == std::chrono::month(5));
	assert(std::chrono::June      == std::chrono::month(6));
	assert(std::chrono::July      == std::chrono::month(7));
	assert(std::chrono::August    == std::chrono::month(8));
	assert(std::chrono::September == std::chrono::month(9));
	assert(std::chrono::October   == std::chrono::month(10));
	assert(std::chrono::November  == std::chrono::month(11));
	assert(std::chrono::December  == std::chrono::month(12));

    std::chrono::year y1 = 2018y;
    assert (y1 == std::chrono::year(2018));
//  No conversion from floating point for years
#endif
}
