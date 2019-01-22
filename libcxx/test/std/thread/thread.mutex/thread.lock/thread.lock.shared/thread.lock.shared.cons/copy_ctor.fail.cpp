//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11
// <shared_mutex>

// template <class Mutex> class shared_lock;

// shared_lock(shared_lock const&) = delete;

#include <shared_mutex>

std::shared_timed_mutex m;

int main()
{
    std::shared_lock<std::shared_timed_mutex> lk0(m);
    std::shared_lock<std::shared_timed_mutex> lk = lk0;
}
