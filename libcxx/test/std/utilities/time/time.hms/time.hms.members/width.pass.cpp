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
// {
// public:
//     static unsigned constexpr fractional_width = see below;
//     using precision                            = see below;
//
//	fractional_width is the number of fractional decimal digits represented by precision.
//  fractional_width has the value of the smallest possible integer in the range [0, 18]
//    such that precision will exactly represent all values of Duration.
//  If no such value of fractional_width exists, then fractional_width is 6.


#include <chrono>
#include <cassert>

#include "test_macros.h"

template <typename Duration, unsigned width>
constexpr bool check_width()
{
	using HMS = std::chrono::hh_mm_ss<Duration>;
	return HMS::fractional_width == width;
}

int main(int, char**)
{
	using microfortnights = std::chrono::duration<int, std::ratio<756, 625>>;

	static_assert( check_width<std::chrono::hours,                               0>(), "");
	static_assert( check_width<std::chrono::minutes,                             0>(), "");
	static_assert( check_width<std::chrono::seconds,                             0>(), "");
	static_assert( check_width<std::chrono::milliseconds,                        3>(), "");
	static_assert( check_width<std::chrono::microseconds,                        6>(), "");
	static_assert( check_width<std::chrono::nanoseconds,                         9>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   2>>, 1>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   3>>, 6>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   4>>, 2>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   5>>, 1>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   6>>, 6>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   7>>, 6>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   8>>, 3>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,   9>>, 6>(), "");
	static_assert( check_width<std::chrono::duration<int, std::ratio<  1,  10>>, 1>(), "");
	static_assert( check_width<microfortnights,                                  4>(), "");

	return 0;
}
