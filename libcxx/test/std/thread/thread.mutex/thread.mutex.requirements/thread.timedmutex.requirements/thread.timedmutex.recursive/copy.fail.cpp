//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// class recursive_timed_mutex;

// recursive_timed_mutex(const recursive_timed_mutex&) = delete;

#include <mutex>

int main()
{
    std::recursive_timed_mutex m0;
    std::recursive_timed_mutex m1(m0);
}
