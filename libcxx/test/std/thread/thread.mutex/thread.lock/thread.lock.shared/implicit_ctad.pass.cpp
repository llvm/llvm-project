//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++98, c++03, c++11, c++14

// <shared_mutex>

// shared_lock

// Make sure that the implicitly-generated CTAD works.

#include <shared_mutex>

#include "test_macros.h"

int main(int, char**) {
  std::shared_mutex mutex;
  {
    std::shared_lock lock(mutex);
    ASSERT_SAME_TYPE(decltype(lock), std::shared_lock<std::shared_mutex>);
  }

  return 0;
}
