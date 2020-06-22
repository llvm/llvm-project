//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class month_day_last;

// constexpr bool operator==(const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() == y.month()
//
// constexpr bool operator< (const month_day& x, const month_day& y) noexcept;
//   Returns: x.month() < y.month()


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"

int main(int, char**)
{
    using month          = std::chrono::month;
    using month_day_last = std::chrono::month_day_last;

    AssertComparisons6AreNoexcept<month_day_last>();
    AssertComparisons6ReturnBool<month_day_last>();

    static_assert( testComparisons6Values<month_day_last>(month{1}, month{1}), "");
    static_assert( testComparisons6Values<month_day_last>(month{1}, month{2}), "");

//  same day, different months
    for (unsigned i = 1; i < 12; ++i)
        for (unsigned j = 1; j < 12; ++j)
            assert((testComparisons6Values<month_day_last>(month{i}, month{j})));

  return 0;
}
