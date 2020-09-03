//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <chrono>

// using months = duration<signed integer type of at least 20 bits, ratio_divide<years::period, ratio<12>>>;


#include <chrono>
#include <type_traits>
#include <limits>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::chrono::months D;
    typedef D::rep Rep;
    typedef D::period Period;
    static_assert(std::is_signed<Rep>::value, "");
    static_assert(std::is_integral<Rep>::value, "");
    static_assert(std::numeric_limits<Rep>::digits >= 20, "");
    static_assert(std::is_same_v<Period, std::ratio_divide<std::chrono::years::period, std::ratio<12>>>, "");

  return 0;
}
