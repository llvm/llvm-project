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

// constexpr bool ok() const noexcept;
//  Returns: If any of y_.ok(), m_.ok(), or wdi_.ok() is false, returns false.
//           Otherwise, if *this represents a valid date, returns true.
//           Otherwise, returns false.

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

    constexpr month January     = std::chrono::January;
    constexpr weekday Monday    = std::chrono::Monday;
    constexpr weekday Tuesday   = std::chrono::Tuesday;
    constexpr weekday Wednesday = std::chrono::Wednesday;
    constexpr weekday Thursday  = std::chrono::Thursday;
    constexpr weekday Friday    = std::chrono::Friday;
    constexpr weekday Saturday  = std::chrono::Saturday;
    constexpr weekday Sunday    = std::chrono::Sunday;

    ASSERT_NOEXCEPT(                std::declval<const year_month_weekday>().ok());
    ASSERT_SAME_TYPE(bool, decltype(std::declval<const year_month_weekday>().ok()));

    static_assert(!year_month_weekday{}.ok(), "");

    static_assert(!year_month_weekday{year{-32768}, month{}, weekday_indexed{}}.ok(),           ""); // All three bad

    static_assert(!year_month_weekday{year{-32768}, January, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad year
    static_assert(!year_month_weekday{year{2019},   month{}, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad month
    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{} }.ok(),          ""); // Bad day

    static_assert(!year_month_weekday{year{-32768}, month{}, weekday_indexed{Tuesday, 1}}.ok(), ""); // Bad year & month
    static_assert(!year_month_weekday{year{2019},   month{}, weekday_indexed{} }.ok(),          ""); // Bad month & day
    static_assert(!year_month_weekday{year{-32768}, January, weekday_indexed{} }.ok(),          ""); // Bad year & day

    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Tuesday, static_cast<unsigned>(-1)}}.ok(), ""); // Bad index.
    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Wednesday, 0}}.ok(), "");                       // Bad index.

    static_assert( year_month_weekday{year{2019},   January, weekday_indexed{Tuesday, 1}}.ok(), ""); // All OK
    static_assert( year_month_weekday{year{2019},   January, weekday_indexed{Tuesday, 4}}.ok(), ""); // All OK

    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Monday, 5}}.ok(),    ""); // Bad index
    static_assert( year_month_weekday{year{2019},   January, weekday_indexed{Tuesday, 5}}.ok(),   ""); // All OK
    static_assert( year_month_weekday{year{2019},   January, weekday_indexed{Wednesday, 5}}.ok(), ""); // All OK
    static_assert( year_month_weekday{year{2019},   January, weekday_indexed{Thursday, 5}}.ok(),  ""); // All OK
    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Friday, 5}}.ok(),    ""); // Bad index
    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Saturday, 5}}.ok(),  ""); // Bad index
    static_assert(!year_month_weekday{year{2019},   January, weekday_indexed{Sunday, 5}}.ok(),    ""); // Bad index

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday ym{year{2019}, January, weekday_indexed{Tuesday, i}};
        assert((ym.ok() == weekday_indexed{Tuesday, i}.ok()));
    }

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday ym{year{2019}, January, weekday_indexed{weekday{i}, 1}};
        assert((ym.ok() == weekday_indexed{weekday{i}, 1}.ok()));
    }

    for (unsigned i = 0; i <= 50; ++i)
    {
        year_month_weekday ym{year{2019}, month{i}, weekday_indexed{Tuesday, 1}};
        assert((ym.ok() == month{i}.ok()));
    }

    const int ymax = static_cast<int>(year::max());
    for (int i = ymax - 100; i <= ymax + 100; ++i)
    {
        year_month_weekday ym{year{i}, January, weekday_indexed{Tuesday, 1}};
        assert((ym.ok() == year{i}.ok()));
    }

  return 0;
}
