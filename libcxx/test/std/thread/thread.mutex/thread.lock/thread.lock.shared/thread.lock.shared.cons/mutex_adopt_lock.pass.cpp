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

// shared_lock(mutex_type& m, adopt_lock_t);

#include <shared_mutex>
#include <cassert>
#include <mutex>
#include "nasty_containers.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::shared_timed_mutex M;
    M m;
    m.lock();
    std::unique_lock<M> lk(m, std::adopt_lock);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    }
    {
    typedef nasty_mutex M;
    M m;
    m.lock();
    std::unique_lock<M> lk(m, std::adopt_lock);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    }

  return 0;
}
