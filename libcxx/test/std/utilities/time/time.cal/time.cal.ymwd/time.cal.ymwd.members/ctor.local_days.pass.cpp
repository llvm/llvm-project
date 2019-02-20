//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// XFAIL: *

// <chrono>
// class year_month_weekday;

//  explicit constexpr year_month_weekday(const local_days& dp) noexcept;
//
//
//  Effects:  Constructs an object of type year_month_weekday that corresponds
//                to the date represented by dp
//
//  Remarks: Equivalent to constructing with sys_days{dp.time_since_epoch()}.
//
//  constexpr chrono::year   year() const noexcept;
//  constexpr chrono::month month() const noexcept;
//  constexpr chrono::day     day() const noexcept;
//  constexpr bool             ok() const noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using year           = std::chrono::year;
    using month          = std::chrono::month;
    using day            = std::chrono::day;
//  using local_days     = std::chrono::local_days;
    using year_month_weekday = std::chrono::year_month_weekday;

//  ASSERT_NOEXCEPT(year_month_weekday{std::declval<const local_days>()});
    assert(false);
}
