//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// <chrono>

// constexpr hours make24(const hours& h, bool is_pm) noexcept;
//   Returns: If is_pm is false, returns the 24-hour equivalent of h in the range [0h, 11h],
//       assuming h represents an ante meridiem hour.
//     Else returns the 24-hour equivalent of h in the range [12h, 23h],
//       assuming h represents a post meridiem hour.
//     If h is not in the range [1h, 12h], the value returned is unspecified.

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using hours = std::chrono::hours;
    ASSERT_SAME_TYPE(hours, decltype(std::chrono::make24(std::declval<hours>(), false)));
    ASSERT_NOEXCEPT(                 std::chrono::make24(std::declval<hours>(), false));

    static_assert( std::chrono::make24(hours( 1), false) == hours( 1), "");
    static_assert( std::chrono::make24(hours(11), false) == hours(11), "");
    static_assert( std::chrono::make24(hours(12), false) == hours( 0), "");
    static_assert( std::chrono::make24(hours( 1), true)  == hours(13), "");
    static_assert( std::chrono::make24(hours(11), true)  == hours(23), "");
    static_assert( std::chrono::make24(hours(12), true)  == hours(12), "");

    for (int i = 1; i < 11; ++i)
    {
        assert((std::chrono::make24(hours(i), false)) == hours(i));
        assert((std::chrono::make24(hours(i), true))  == hours(12+i));
    }
    assert((std::chrono::make24(hours(12), false)) == hours( 0));
    assert((std::chrono::make24(hours(12), true))  == hours(12));

    return 0;
}
