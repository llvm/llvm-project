//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11
// XFAIL: dylib-has-no-shared_mutex

// <shared_mutex>

// template <class Mutex> class shared_lock;

// shared_lock(mutex_type& m, defer_lock_t);

#include <shared_mutex>
#include <cassert>
#include "nasty_containers.h"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::shared_timed_mutex M;
    M m;
    std::unique_lock<M> lk(m, std::defer_lock);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m;
    std::unique_lock<M> lk(m, std::defer_lock);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == false);
    }

  return 0;
}
