//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread

// thread(const thread&) = delete;

#include <thread>

int main(int, char**)
{
    std::thread t0; (void)t0;
    std::thread t1(t0); (void)t1;
    return 0;
}
