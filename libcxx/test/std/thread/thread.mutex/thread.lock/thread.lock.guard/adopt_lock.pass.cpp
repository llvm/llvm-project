//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <mutex>

// template <class Mutex> class lock_guard;

// lock_guard(mutex_type& m, adopt_lock_t);

#include <mutex>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

struct MyMutex {
  bool locked = false;

  MyMutex() = default;
  ~MyMutex() { assert(!locked); }

  void lock() {
    assert(!locked);
    locked = true;
  }
  void unlock() {
    assert(locked);
    locked = false;
  }

  MyMutex(MyMutex const&)            = delete;
  MyMutex& operator=(MyMutex const&) = delete;
};

int main(int, char**) {
  MyMutex m;
  {
    m.lock();
    std::lock_guard<MyMutex> lg(m, std::adopt_lock);
    assert(m.locked);
  }

  m.lock();
  m.unlock();

  return 0;
}
