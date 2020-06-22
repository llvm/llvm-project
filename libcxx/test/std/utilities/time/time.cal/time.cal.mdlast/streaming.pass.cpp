//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: *

// <chrono>
// class month_day_last;
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day_last& mdl);
//
//     Returns: os << mdl.month() << "/last".


#include <chrono>
#include <type_traits>
#include <cassert>
#include <iostream>

#include "test_macros.h"

int main(int, char**)
{
    using month_day_last = std::chrono::month_day_last;
    using month          = std::chrono::month;
    std::cout << month_day_last{month{1}};

  return 0;
}
