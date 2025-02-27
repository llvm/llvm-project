//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month_weekday_last;

// constexpr chrono::weekday_last weekday_last() const noexcept;
//  Returns: wdl_

#include <chrono>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"

int main(int, char**)
{
    using month              = std::chrono::month;
    using weekday            = std::chrono::weekday;
    using weekday_last       = std::chrono::weekday_last;
    using month_weekday_last = std::chrono::month_weekday_last;

    constexpr month January            = std::chrono::January;
    constexpr weekday Tuesday          = std::chrono::Tuesday;
    constexpr weekday_last lastTuesday = weekday_last{Tuesday};

    ASSERT_NOEXCEPT(                        std::declval<const month_weekday_last>().weekday_last());
    ASSERT_SAME_TYPE(weekday_last, decltype(std::declval<const month_weekday_last>().weekday_last()));

    static_assert( month_weekday_last{month{}, lastTuesday}.weekday_last() == lastTuesday, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        month_weekday_last mdl(January, weekday_last{weekday{i}});
        assert( mdl.weekday_last().weekday().c_encoding() == (i == 7 ? 0 : i));
    }

  return 0;
}
