//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <chrono>

// constexpr bool is_pm(const hours& h) noexcept;
//   Returns: 12h <= h && h <= 23

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using hours = std::chrono::hours;
    ASSERT_SAME_TYPE(bool, decltype(std::chrono::is_pm(std::declval<hours>())));
    ASSERT_NOEXCEPT(                std::chrono::is_pm(std::declval<hours>()));

    static_assert(!std::chrono::is_pm(hours( 0)), "");
    static_assert(!std::chrono::is_pm(hours(11)), "");
    static_assert( std::chrono::is_pm(hours(12)), "");
    static_assert( std::chrono::is_pm(hours(23)), "");

    for (int i = 0; i < 12; ++i)
        assert(!std::chrono::is_pm(hours(i)));
    for (int i = 12; i < 24; ++i)
        assert( std::chrono::is_pm(hours(i)));

    return 0;
}
