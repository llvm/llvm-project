//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17

// <chrono>

// class month_weekday_last;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    using month_weekday_last = std::chrono::month_weekday_last;

    static_assert(std::is_trivially_copyable_v<month_weekday_last>, "");
    static_assert(std::is_standard_layout_v<month_weekday_last>, "");

  return 0;
}
