//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type* release() noexcept;

#include <cassert>
#include <mutex>

#include "test_macros.h"
#include "../types.h"

int MyCountingMutex::lock_count   = 0;
int MyCountingMutex::unlock_count = 0;

MyCountingMutex m;

int main(int, char**) {
  std::unique_lock<MyCountingMutex> lk(m);
  assert(lk.mutex() == &m);
  assert(lk.owns_lock() == true);
  assert(MyCountingMutex::lock_count == 1);
  assert(MyCountingMutex::unlock_count == 0);
  assert(lk.release() == &m);
  assert(lk.mutex() == nullptr);
  assert(lk.owns_lock() == false);
  assert(MyCountingMutex::lock_count == 1);
  assert(MyCountingMutex::unlock_count == 0);

  return 0;
}
