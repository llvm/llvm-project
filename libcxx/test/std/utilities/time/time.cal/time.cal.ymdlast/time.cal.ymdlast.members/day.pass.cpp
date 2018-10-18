//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: *

// <chrono>
// class year_month_day_last;

// constexpr chrono::day day() const noexcept;
//  Returns: wd_

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using year                = std::chrono::year;
    using month               = std::chrono::month;
    using day                 = std::chrono::day;
    using month_day_last      = std::chrono::month_day_last;
    using year_month_day_last = std::chrono::year_month_day_last;

//  TODO: wait for calendar
//     ASSERT_NOEXCEPT(               std::declval<const year_month_day_last>().day());
//     ASSERT_SAME_TYPE(day, decltype(std::declval<const year_month_day_last>().day()));
// 
//     static_assert( year_month_day_last{}.day() == day{}, "");

    for (unsigned i = 1; i <= 12; ++i)
    {
        year_month_day_last ymd(year{1234}, month_day_last{month{i}});
        assert( static_cast<unsigned>(ymd.day()) == i);
    }
}
