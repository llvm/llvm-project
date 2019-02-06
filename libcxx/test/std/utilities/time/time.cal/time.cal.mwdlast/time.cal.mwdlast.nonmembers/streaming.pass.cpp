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
// class month_weekday_last;
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_weekday_last& mdl);
//
//     Returns: os << mdl.month() << "/last".


#include <chrono>
#include <type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"

int main()
{
    using month_weekday_last = std::chrono::month_weekday_last;
    using month              = std::chrono::month;
    using weekday            = std::chrono::weekday;
    using weekday_last       = std::chrono::weekday_last;

    std::cout << month_weekday_last{month{1}, weekday_last{weekday{3}}};
}
