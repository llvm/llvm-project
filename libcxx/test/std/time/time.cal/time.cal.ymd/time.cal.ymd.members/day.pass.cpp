//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_day;

// constexpr chrono::day day() const noexcept;
//  Returns: wd_

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    using year           = std::chrono::year;
    using month          = std::chrono::month;
    using day            = std::chrono::day;
    using year_month_day = std::chrono::year_month_day;

    ASSERT_NOEXCEPT(               std::declval<const year_month_day>().day());
    ASSERT_SAME_TYPE(day, decltype(std::declval<const year_month_day>().day()));

    static_assert( year_month_day{}.day() == day{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month_day ymd(year{1234}, month{2}, day{i});
        assert( static_cast<unsigned>(ymd.day()) == i);
    }

  return 0;
}
