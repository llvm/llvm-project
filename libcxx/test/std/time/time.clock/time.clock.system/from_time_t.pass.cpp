//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// "unable to find library from dependent library specifier: rt"
// XFAIL: LIBCXX-PICOLIBC-FIXME

// <chrono>

// system_clock

// static time_point from_time_t(time_t t);

#include <chrono>
#include <ctime>

#include "test_macros.h"

int main(int, char**)
{
    typedef std::chrono::system_clock C;
    C::time_point t1 = C::from_time_t(C::to_time_t(C::now()));
    ((void)t1);

  return 0;
}
