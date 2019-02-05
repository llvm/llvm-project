//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class month_weekday_last;

// constexpr chrono::month month() const noexcept;
//  Returns: m_

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main()
{
    using month              = std::chrono::month;
    using weekday            = std::chrono::weekday;
    using weekday_last       = std::chrono::weekday_last;
    using month_weekday_last = std::chrono::month_weekday_last;

    constexpr weekday Tuesday = std::chrono::Tuesday;

    ASSERT_NOEXCEPT(                 std::declval<const month_weekday_last>().month());
    ASSERT_SAME_TYPE(month, decltype(std::declval<const month_weekday_last>().month()));

    static_assert( month_weekday_last{month{}, weekday_last{Tuesday}}.month() == month{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        month_weekday_last mdl(month{i}, weekday_last{Tuesday});
        assert( static_cast<unsigned>(mdl.month()) == i);
    }
}
