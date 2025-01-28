//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month_weekday;

// constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  Returns: wd_

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    using year               = std::chrono::year;
    using month              = std::chrono::month;
    using weekday            = std::chrono::weekday;
    using weekday_indexed    = std::chrono::weekday_indexed;
    using year_month_weekday = std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(                           std::declval<const year_month_weekday>().weekday_indexed());
    ASSERT_SAME_TYPE(weekday_indexed, decltype(std::declval<const year_month_weekday>().weekday_indexed()));

    static_assert( year_month_weekday{}.weekday_indexed() == weekday_indexed{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month_weekday ymwd0(year{1234}, month{2}, weekday_indexed{weekday{i}, 1});
        assert( ymwd0.weekday_indexed().weekday().c_encoding() == (i == 7 ? 0 : i));
        assert( static_cast<unsigned>(ymwd0.weekday_indexed().index()) == 1);

        year_month_weekday ymwd1(year{1234}, month{2}, weekday_indexed{weekday{2}, i});
        assert( ymwd1.weekday_indexed().weekday().c_encoding() == 2);
        assert( static_cast<unsigned>(ymwd1.weekday_indexed().index()) == i);
    }

  return 0;
}
