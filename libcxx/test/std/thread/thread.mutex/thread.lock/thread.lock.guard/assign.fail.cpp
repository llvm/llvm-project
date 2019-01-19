//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard& operator=(lock_guard const&) = delete;

#include <mutex>

int main()
{
    std::mutex m0;
    std::mutex m1;
    std::lock_guard<std::mutex> lg0(m0);
    std::lock_guard<std::mutex> lg(m1);
    lg = lg0;
}
