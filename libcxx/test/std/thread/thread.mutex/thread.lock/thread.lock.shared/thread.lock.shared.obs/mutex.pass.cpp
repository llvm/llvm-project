//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// XFAIL: availability-shared_mutex-missing

// <shared_mutex>

// template <class Mutex> class shared_lock;

// mutex_type *mutex() const noexcept;

#include <shared_mutex>
#include <cassert>

#include "test_macros.h"

std::shared_timed_mutex m;

int main(int, char**)
{
    std::shared_lock<std::shared_timed_mutex> lk0;
    assert(lk0.mutex() == nullptr);
    std::shared_lock<std::shared_timed_mutex> lk1(m);
    assert(lk1.mutex() == &m);
    lk1.unlock();
    assert(lk1.mutex() == &m);
    static_assert(noexcept(lk0.mutex()), "mutex() must be noexcept");

  return 0;
}
