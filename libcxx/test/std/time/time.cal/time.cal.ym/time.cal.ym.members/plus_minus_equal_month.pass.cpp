//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>
// class year_month;

// constexpr year_month& operator+=(const months& d) noexcept;
// constexpr year_month& operator-=(const months& d) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

template <typename D, typename Ds>
constexpr bool testConstexpr(D d1)
{
    if (static_cast<unsigned>((d1          ).month()) !=  1) return false;
    if (static_cast<unsigned>((d1 += Ds{ 1}).month()) !=  2) return false;
    if (static_cast<unsigned>((d1 += Ds{ 2}).month()) !=  4) return false;
    if (static_cast<unsigned>((d1 += Ds{12}).month()) !=  4) return false;
    if (static_cast<unsigned>((d1 -= Ds{ 1}).month()) !=  3) return false;
    if (static_cast<unsigned>((d1 -= Ds{ 2}).month()) !=  1) return false;
    if (static_cast<unsigned>((d1 -= Ds{12}).month()) !=  1) return false;
    return true;
}

int main(int, char**)
{
    using month      = std::chrono::month;
    using months     = std::chrono::months;
    using year       = std::chrono::year;
    using year_month = std::chrono::year_month;

    ASSERT_NOEXCEPT(                       std::declval<year_month&>() += std::declval<months>());
    ASSERT_SAME_TYPE(year_month&, decltype(std::declval<year_month&>() += std::declval<months>()));

    ASSERT_NOEXCEPT(                       std::declval<year_month&>() -= std::declval<months>());
    ASSERT_SAME_TYPE(year_month&, decltype(std::declval<year_month&>() -= std::declval<months>()));

    static_assert(testConstexpr<year_month, months>(year_month{year{1234}, month{1}}), "");

    for (unsigned i = 0; i <= 10; ++i)
    {
        year y{1234};
        year_month ym(y, month{i});
        assert(static_cast<unsigned>((ym += months{2}).month()) == i + 2);
        assert(ym.year() == y);
        assert(static_cast<unsigned>((ym             ).month()) == i + 2);
        assert(ym.year() == y);
        assert(static_cast<unsigned>((ym -= months{1}).month()) == i + 1);
        assert(ym.year() == y);
        assert(static_cast<unsigned>((ym             ).month()) == i + 1);
        assert(ym.year() == y);
    }

  return 0;
}
