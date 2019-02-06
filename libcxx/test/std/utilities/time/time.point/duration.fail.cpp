//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>

// time_point

// Duration shall be an instance of duration.

#include <chrono>

int main()
{
    typedef std::chrono::time_point<std::chrono::system_clock, int> T;
    T t;
}
