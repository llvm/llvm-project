//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type *mutex() const;

#include <mutex>
#include <cassert>

#include "test_macros.h"

std::mutex m;

int main(int, char**)
{
    std::unique_lock<std::mutex> lk0;
    assert(lk0.mutex() == nullptr);
    std::unique_lock<std::mutex> lk1(m);
    assert(lk1.mutex() == &m);
    lk1.unlock();
    assert(lk1.mutex() == &m);

  return 0;
}
