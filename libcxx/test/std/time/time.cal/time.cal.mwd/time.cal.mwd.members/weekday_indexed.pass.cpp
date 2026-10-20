//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month_weekday;

// constexpr chrono::weekday_indexed weekday_indexed() const noexcept;
//  Returns: wdi_

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    using month_weekday   = std::chrono::month_weekday;
    using month           = std::chrono::month;
    using weekday         = std::chrono::weekday;
    using weekday_indexed = std::chrono::weekday_indexed;

    constexpr weekday Sunday = std::chrono::Sunday;

    ASSERT_NOEXCEPT(                           std::declval<const month_weekday>().weekday_indexed());
    ASSERT_SAME_TYPE(weekday_indexed, decltype(std::declval<const month_weekday>().weekday_indexed()));

    static_assert( month_weekday{month{}, weekday_indexed{}}.weekday_indexed() == weekday_indexed{}, "");

    for (unsigned i = 1; i <= 10; ++i)
    {
        month_weekday md(std::chrono::March, weekday_indexed{Sunday, i});
        assert( static_cast<unsigned>(md.weekday_indexed().weekday() == Sunday));
        assert( static_cast<unsigned>(md.weekday_indexed().index() == i));
    }

  return 0;
}
