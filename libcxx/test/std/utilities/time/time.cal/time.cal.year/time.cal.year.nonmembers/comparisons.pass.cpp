//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year;

// constexpr bool operator==(const year& x, const year& y) noexcept;
// constexpr bool operator!=(const year& x, const year& y) noexcept;
// constexpr bool operator< (const year& x, const year& y) noexcept;
// constexpr bool operator> (const year& x, const year& y) noexcept;
// constexpr bool operator<=(const year& x, const year& y) noexcept;
// constexpr bool operator>=(const year& x, const year& y) noexcept;


#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_comparisons.h"


int main(int, char**)
{
    using year = std::chrono::year;

    AssertComparisons6AreNoexcept<year>();
    AssertComparisons6ReturnBool<year>();

    static_assert(testComparisons6Values<year>(0,0), "");
    static_assert(testComparisons6Values<year>(0,1), "");

    //  Some 'ok' values as well
    static_assert(testComparisons6Values<year>( 5, 5), "");
    static_assert(testComparisons6Values<year>( 5,10), "");

    for (int i = 1; i < 10; ++i)
        for (int j = 1; j < 10; ++j)
            assert(testComparisons6Values<year>(i, j));

    return 0;
}
