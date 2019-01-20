//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// static constexpr time_point min(); // noexcept after C++17

#include <chrono>
#include <cassert>

#include "test_macros.h"

int main()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds Duration;
    typedef std::chrono::time_point<Clock, Duration> TP;
    LIBCPP_ASSERT_NOEXCEPT(TP::max());
#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(       TP::max());
#endif
    assert(TP::min() == TP(Duration::min()));
}
