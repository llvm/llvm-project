//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type *mutex() const;

#include <cassert>
#include <memory>
#include <mutex>

#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex mux;
  std::unique_lock<checking_mutex> lock0;
  assert(lock0.mutex() == nullptr);
  std::unique_lock<checking_mutex> lock1(mux);
  assert(lock1.mutex() == std::addressof(mux));
  lock1.unlock();
  assert(lock1.mutex() == std::addressof(mux));

  return 0;
}
