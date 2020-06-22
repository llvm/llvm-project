//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class weekday_indexed;

// constexpr chrono::weekday weekday() const noexcept;
//  Returns: wd_

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using weekday         = std::chrono::weekday;
    using weekday_indexed = std::chrono::weekday_indexed;

    ASSERT_NOEXCEPT(                                std::declval<const weekday_indexed>().weekday());
    ASSERT_SAME_TYPE(std::chrono::weekday, decltype(std::declval<const weekday_indexed>().weekday()));

    static_assert( weekday_indexed{}.weekday() == weekday{},                                   "");
    static_assert( weekday_indexed{std::chrono::Tuesday, 0}.weekday() == std::chrono::Tuesday, "");

    for (unsigned i = 0; i <= 6; ++i)
    {
        weekday_indexed wdi(weekday{i}, 2);
        assert( wdi.weekday().c_encoding() == i);
    }

  return 0;
}
