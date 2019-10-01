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
// class month_day;

// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const month_day& md);
//
//     Returns: os << md.month() << '/' << md.day().
//
// template<class charT, class traits>
//     basic_ostream<charT, traits>&
//     to_stream(basic_ostream<charT, traits>& os, const charT* fmt, const month_day& md);
//
// Effects: Streams md into os using the format specified by the NTCTS fmt.
//          fmt encoding follows the rules specified in 25.11.


#include <chrono>
#include <type_traits>
#include <cassert>
#include <iostream>
#include "test_macros.h"

int main(int, char**)
{
    using month_day = std::chrono::month_day;
    using month     = std::chrono::month;
    using day       = std::chrono::day;
    std::cout << month_day{month{1}, day{1}};

  return 0;
}
