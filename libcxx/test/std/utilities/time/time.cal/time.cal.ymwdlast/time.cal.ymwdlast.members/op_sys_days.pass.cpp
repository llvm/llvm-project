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

// constexpr operator sys_days() const noexcept;
//  Returns: sys_days{year()/month()/day()}.

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
    using sys_days            = std::chrono::sys_days;

    ASSERT_NOEXCEPT(                static_cast<sys_days>(std::declval<const year_month_day_last>().year()));
    ASSERT_SAME_TYPE(year, decltype(static_cast<sys_days>(std::declval<const year_month_day_last>().year()));

}
