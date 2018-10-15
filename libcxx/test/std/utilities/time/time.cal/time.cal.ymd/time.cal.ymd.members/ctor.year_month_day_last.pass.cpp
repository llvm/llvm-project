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
// class year_month_day;

//  constexpr year_month_day(const year_month_day_last& ymdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_day by initializing
//              y_ with ymdl.year(), m_ with ymdl.month(), and d_ with ymdl.day().
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
    using year                = std::chrono::year;
    using month               = std::chrono::month;
    using day                 = std::chrono::day;
//  using year_month_day_last = std::chrono::year_month_day_last;
    using year_month_day      = std::chrono::year_month_day;

//  ASSERT_NOEXCEPT(year_month_day{std::declval<const year_month_day_last>()});
    assert(false);
    
}
