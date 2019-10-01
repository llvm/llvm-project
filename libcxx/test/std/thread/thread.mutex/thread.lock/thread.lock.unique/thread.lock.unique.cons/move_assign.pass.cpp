//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, c++98, c++03

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock& operator=(unique_lock&& u);

#include <mutex>
#include <cassert>
#include "nasty_containers.hpp"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::mutex M;
    M m0;
    M m1;
    std::unique_lock<M> lk0(m0);
    std::unique_lock<M> lk1(m1);
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
    std::unique_lock<M> lk0(m0);
    std::unique_lock<M> lk1(m1);
    lk1 = std::move(lk0);
    assert(lk1.mutex() == std::addressof(m0));
    assert(lk1.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }

  return 0;
}
