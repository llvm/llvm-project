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

// constexpr year operator""y(unsigned long long y) noexcept;

#include <chrono>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    using namespace std::chrono;
    ASSERT_NOEXCEPT(4y);

    static_assert( 2017y == year(2017), "");
    year y1 = 2018y;
    assert (y1 == year(2018));
    }

    {
    using namespace std::literals;
    ASSERT_NOEXCEPT(4d);

    static_assert( 2017y == std::chrono::year(2017), "");

    std::chrono::year y1 = 2020y;
    assert (y1 == std::chrono::year(2020));
    }

  return 0;
}
