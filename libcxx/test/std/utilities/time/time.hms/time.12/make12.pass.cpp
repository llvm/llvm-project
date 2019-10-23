//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <chrono>

// constexpr hours make12(const hours& h) noexcept;
//   Returns: The 12-hour equivalent of h in the range [1h, 12h].
//     If h is not in the range [0h, 23h], the value returned is unspecified.

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using hours = std::chrono::hours;
    ASSERT_SAME_TYPE(hours, decltype(std::chrono::make12(std::declval<hours>())));
    ASSERT_NOEXCEPT(                 std::chrono::make12(std::declval<hours>()));

    static_assert( std::chrono::make12(hours( 0)) == hours(12), "");
    static_assert( std::chrono::make12(hours(11)) == hours(11), "");
    static_assert( std::chrono::make12(hours(12)) == hours(12), "");
    static_assert( std::chrono::make12(hours(23)) == hours(11), "");

    assert( std::chrono::make12(hours(0)) == hours(12));
    for (int i = 1; i < 13; ++i)
        assert( std::chrono::make12(hours(i)) == hours(i));
    for (int i = 13; i < 24; ++i)
        assert( std::chrono::make12(hours(i)) == hours(i-12));

    return 0;
}
