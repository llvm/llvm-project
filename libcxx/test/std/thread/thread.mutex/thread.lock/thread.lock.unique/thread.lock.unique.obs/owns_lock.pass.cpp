//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// bool owns_lock() const;

#include <cassert>
#include <mutex>

#include "test_macros.h"
#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex mux;
  std::unique_lock<checking_mutex> lock0;
  assert(!lock0.owns_lock());
  std::unique_lock<checking_mutex> lock1(mux);
  assert(lock1.owns_lock());
  lock1.unlock();
  assert(!lock1.owns_lock());

  return 0;
}
