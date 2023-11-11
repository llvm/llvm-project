//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: no-threads
// XFAIL: availability-shared_resource-missing

#include <__utility/shared_resource.h>
#include <mutex>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  int a = 0;
  int b = 0;
  int c = 0;

  std::mutex& a_mutex = std::__shared_resource_inc_reference(&a);
  std::mutex& b_mutex = std::__shared_resource_inc_reference(&b);
  std::__shared_resource_inc_reference(&c);

  { // lock all three objects, all objects have an unique address so no dead locks.
    std::lock_guard a_lock{a_mutex};
    std::lock_guard b_lock{b_mutex};
    std::lock_guard c_lock = std::__shared_resource_get_lock(&c);
  }
  { // Test the lock of a locks the mutex returned by inc.
    std::lock_guard a_lock = std::__shared_resource_get_lock(&a);
    assert(!a_mutex.try_lock());

    std::lock_guard b_lock{b_mutex};
    std::lock_guard c_lock = std::__shared_resource_get_lock(&c);
  }

  std::__shared_resource_dec_reference(&a);
  std::__shared_resource_dec_reference(&b);
  std::__shared_resource_dec_reference(&c);

  return 0;
}
