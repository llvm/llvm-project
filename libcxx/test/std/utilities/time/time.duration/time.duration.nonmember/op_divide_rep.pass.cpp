//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// template <class Rep1, class Period, class Rep2>
//   constexpr
//   duration<typename common_type<Rep1, Rep2>::type, Period>
//   operator/(const duration<Rep1, Period>& d, const Rep2& s);

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
    std::chrono::nanoseconds ns(15);
    ns = ns / 5;
    assert(ns.count() == 3);
    }
#if TEST_STD_VER >= 11
    {
    constexpr std::chrono::nanoseconds ns(15);
    constexpr std::chrono::nanoseconds ns2 = ns / 5;
    static_assert(ns2.count() == 3, "");
    }
#endif

  return 0;
}
