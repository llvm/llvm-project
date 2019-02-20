//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard(lock_guard const&) = delete;

#include <mutex>

int main(int, char**)
{
    std::mutex m;
    std::lock_guard<std::mutex> lg0(m);
    std::lock_guard<std::mutex> lg(lg0);

  return 0;
}
