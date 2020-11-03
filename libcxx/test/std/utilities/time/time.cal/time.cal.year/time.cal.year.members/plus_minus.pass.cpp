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

// constexpr year operator+() const noexcept;
// constexpr year operator-() const noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename Y>
constexpr bool testConstexpr()
{
    Y y1{1};
    if (static_cast<int>(+y1) !=  1) return false;
    if (static_cast<int>(-y1) != -1) return false;
    return true;
}

int main(int, char**)
{
    using year  = std::chrono::year;

    ASSERT_NOEXCEPT(+std::declval<year>());
    ASSERT_NOEXCEPT(-std::declval<year>());

    ASSERT_SAME_TYPE(year, decltype(+std::declval<year>()));
    ASSERT_SAME_TYPE(year, decltype(-std::declval<year>()));

    static_assert(testConstexpr<year>(), "");

    for (int i = 10000; i <= 10020; ++i)
    {
        year yr(i);
        assert(static_cast<int>(+yr) ==  i);
        assert(static_cast<int>(-yr) == -i);
    }

  return 0;
}
