//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  Libc++ adds noexcept to the default constructor to std::chrono::duration
//    when the underlying type is noexcept default constructible. This makes all
//    the standard durations noexcept default constructible.

// typedef duration<long long,         nano> nanoseconds;
// typedef duration<long long,        micro> microseconds;
// typedef duration<long long,        milli> milliseconds;
// typedef duration<long long              > seconds;
// typedef duration<     long, ratio<  60> > minutes;
// typedef duration<     long, ratio<3600> > hours;
// 
// #if _LIBCPP_STD_VER > 17
// typedef duration<     int, ratio_multiply<ratio<24>, hours::period>>         days;
// typedef duration<     int, ratio_multiply<ratio<7>,   days::period>>         weeks;
// typedef duration<     int, ratio_multiply<ratio<146097, 400>, days::period>> years;
// typedef duration<     int, ratio_divide<years::period, ratio<12>>>           months;
// #endif

#include <chrono>

#include "test_macros.h"

int main(int, char**) {

    ASSERT_NOEXCEPT(std::chrono::nanoseconds());
    ASSERT_NOEXCEPT(std::chrono::microseconds());
    ASSERT_NOEXCEPT(std::chrono::milliseconds());
    ASSERT_NOEXCEPT(std::chrono::seconds());
    ASSERT_NOEXCEPT(std::chrono::minutes());
    ASSERT_NOEXCEPT(std::chrono::hours());

#if TEST_STD_VER > 17
    ASSERT_NOEXCEPT(std::chrono::days());
    ASSERT_NOEXCEPT(std::chrono::weeks());
    ASSERT_NOEXCEPT(std::chrono::years());
    ASSERT_NOEXCEPT(std::chrono::months());
#endif

    return 0;
}
