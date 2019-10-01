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

// unique_lock(unique_lock&& u);

#include <mutex>
#include <cassert>
#include "nasty_containers.hpp"

#include "test_macros.h"

int main(int, char**)
{
    {
    typedef std::mutex M;
    M m;
    std::unique_lock<M> lk0(m);
    std::unique_lock<M> lk = std::move(lk0);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }
    {
    typedef nasty_mutex M;
    M m;
    std::unique_lock<M> lk0(m);
    std::unique_lock<M> lk = std::move(lk0);
    assert(lk.mutex() == std::addressof(m));
    assert(lk.owns_lock() == true);
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);
    }

  return 0;
}
