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
//   precision is duration<common_type_t<Duration::rep, seconds::rep>,
//                                 ratio<1, 10^^fractional_width>>

#include <chrono>
#include <cassert>

#include "test_macros.h"

constexpr unsigned long long powers[] = {
	1ULL,
	10ULL,
	100ULL,
	1000ULL,
	10000ULL,
	100000ULL,
	1000000ULL,
	10000000ULL,
	100000000ULL,
	1000000000ULL,
	10000000000ULL,
	100000000000ULL,
	1000000000000ULL,
	10000000000000ULL,
	100000000000000ULL,
	1000000000000000ULL,
	10000000000000000ULL,
	100000000000000000ULL,
	1000000000000000000ULL,
	10000000000000000000ULL
};

template <typename Duration, unsigned width>
constexpr bool check_precision()
{
	using HMS = std::chrono::hh_mm_ss<Duration>;
	using CT  = std::common_type_t<typename Duration::rep, std::chrono::seconds::rep>;
	using Pre = std::chrono::duration<CT, std::ratio<1, powers[width]>>;
	return std::is_same_v<typename HMS::precision, Pre>;
}

int main(int, char**)
{
	using microfortnights = std::chrono::duration<int, std::ratio<756, 625>>;

	static_assert( check_precision<std::chrono::hours,                               0>(), "");
	static_assert( check_precision<std::chrono::minutes,                             0>(), "");
	static_assert( check_precision<std::chrono::seconds,                             0>(), "");
	static_assert( check_precision<std::chrono::milliseconds,                        3>(), "");
	static_assert( check_precision<std::chrono::microseconds,                        6>(), "");
	static_assert( check_precision<std::chrono::nanoseconds,                         9>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   2>>, 1>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   3>>, 6>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   4>>, 2>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   5>>, 1>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   6>>, 6>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   7>>, 6>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   8>>, 3>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,   9>>, 6>(), "");
	static_assert( check_precision<std::chrono::duration<int, std::ratio<  1,  10>>, 1>(), "");
	static_assert( check_precision<microfortnights,                                  4>(), "");

	return 0;
}
