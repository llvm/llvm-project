//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread

// thread& operator=(thread&& t);

#include <thread>

int main(int, char**)
{
    std::thread t0;
    std::thread t1;
    t0 = t1;
    return 0;
}
