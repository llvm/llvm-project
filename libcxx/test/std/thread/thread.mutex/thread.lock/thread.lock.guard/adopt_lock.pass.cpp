//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard(mutex_type& m, adopt_lock_t);

#include <mutex>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

std::mutex m;

int main()
{
  {
    m.lock();
    std::lock_guard<std::mutex> lg(m, std::adopt_lock);
    assert(m.try_lock() == false);
  }

  m.lock();
  m.unlock();

  return 0;
}
