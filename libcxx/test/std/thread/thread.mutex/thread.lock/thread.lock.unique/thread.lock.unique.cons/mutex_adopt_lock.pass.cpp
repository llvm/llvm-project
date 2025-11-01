//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock(mutex_type& m, adopt_lock_t);

#include <cassert>
#include <memory>
#include <mutex>

#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex m;
  m.lock();
  m.last_try = checking_mutex::none;
  std::unique_lock<checking_mutex> lk(m, std::adopt_lock_t());
  assert(m.last_try == checking_mutex::none);
  assert(lk.mutex() == std::addressof(m));
  assert(lk.owns_lock());

  return 0;
}
