//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <chrono>

// ceil

// template <class Rep, class Period>
//   constexpr duration<Rep, Period> abs(duration<Rep, Period> d)

// This function shall not participate in overload resolution unless numeric_limits<Rep>::is_signed is true.

#include <chrono>

typedef std::chrono::duration<unsigned> unsigned_secs;

int main(int, char**)
{
    std::chrono::abs(unsigned_secs(0));

  return 0;
}
