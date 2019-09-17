//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11
// XFAIL: dylib-has-no-shared_mutex

// <shared_mutex>

// template <class Mutex> class shared_lock;

// shared_lock& operator=(shared_lock&& u);

#include <shared_mutex>
#include <cassert>
#include "nasty_containers.h"

#include "test_macros.h"


int main(int, char**)
{
    {
    typedef std::shared_timed_mutex M;
    M m0;
    M m1;
    std::shared_lock<M> lk0(m0);
    std::shared_lock<M> lk1(m1);
    lk1 = std::move(lk0);
    assert(lk1.mutex() == std::addressof(m0));
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m0;
    M m1;
    std::shared_lock<M> lk0(m0);
    std::shared_lock<M> lk1(m1);
    lk1 = std::move(lk0);
    assert(lk1.mutex() == std::addressof(m0));
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }

  return 0;
}
