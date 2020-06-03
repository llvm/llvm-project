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

// constexpr chrono::weekday weekday() const noexcept;
//  Returns: wdi_.weekday()

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using year               = std::chrono::year;
    using month              = std::chrono::month;
    using weekday            = std::chrono::weekday;
    using weekday_indexed    = std::chrono::weekday_indexed;
    using year_month_weekday = std::chrono::year_month_weekday;

    ASSERT_NOEXCEPT(                   std::declval<const year_month_weekday>().weekday());
    ASSERT_SAME_TYPE(weekday, decltype(std::declval<const year_month_weekday>().weekday()));

    static_assert( year_month_weekday{}.weekday() == weekday{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        year_month_weekday ymwd0(year{1234}, month{2}, weekday_indexed{weekday{i}, 1});
        assert(ymwd0.weekday().c_encoding() == (i == 7 ? 0 : i));
    }

  return 0;
}
