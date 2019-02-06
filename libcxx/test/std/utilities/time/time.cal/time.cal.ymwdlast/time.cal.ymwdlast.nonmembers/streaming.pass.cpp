//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// XFAIL: *

// <chrono>
// class year_month_weekday_last;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const year_month_weekday_last& ymwdl);
//
//   Returns: os << ymwdl.year() << '/' << ymwdl.month() << '/' << ymwdl.weekday_last().


#include <chrono>
#include <type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"

int main()
{
    using year_month_weekday_last = std::chrono::year_month_weekday_last;
    using year                    = std::chrono::year;
    using month                   = std::chrono::month;
    using weekday                 = std::chrono::weekday;
    using weekday_last            = std::chrono::weekday_last;

    std::cout << year_month_weekday_last{year{2018}, month{3}, weekday_last{weekday{4}}};
}
