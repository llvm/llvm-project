//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>
// class month_day;

// constexpr chrono::day day() const noexcept;
//  Returns: d_

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using day       = std::chrono::day;
    using month_day = std::chrono::month_day;

    ASSERT_NOEXCEPT(               std::declval<const month_day>().day());
    ASSERT_SAME_TYPE(day, decltype(std::declval<const month_day>().day()));

    static_assert( month_day{}.day() == day{}, "");

    for (unsigned i = 1; i <= 50; ++i)
    {
        month_day md(std::chrono::March, day{i});
        assert( static_cast<unsigned>(md.day()) == i);
    }

  return 0;
}
