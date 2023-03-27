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

// shared_lock();

#include <shared_mutex>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    std::shared_lock<std::shared_timed_mutex> ul;
    assert(!ul.owns_lock());
    assert(ul.mutex() == nullptr);

  return 0;
}
