//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// system_clock

// time_t to_time_t(const time_point& t);

#include <chrono>
#include <ctime>

int main()
{
    typedef std::chrono::system_clock C;
    std::time_t t1 = C::to_time_t(C::now());
    ((void)t1);
}
