//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// <chrono>

// constexpr bool is_am(const hours& h) noexcept;
//   Returns: 0h <= h && h <= 11h.

#include <chrono>
#include <cassert>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
	using hours = std::chrono::hours;
	ASSERT_SAME_TYPE(bool, decltype(std::chrono::is_am(std::declval<hours>())));
	ASSERT_NOEXCEPT(                std::chrono::is_am(std::declval<hours>()));

	static_assert( std::chrono::is_am(hours( 0)), "");
	static_assert( std::chrono::is_am(hours(11)), "");
	static_assert(!std::chrono::is_am(hours(12)), "");
	static_assert(!std::chrono::is_am(hours(23)), "");

	for (int i = 0; i < 12; ++i)
		assert( std::chrono::is_am(hours(i)));
	for (int i = 12; i < 24; ++i)
		assert(!std::chrono::is_am(hours(i)));

    return 0;
}
