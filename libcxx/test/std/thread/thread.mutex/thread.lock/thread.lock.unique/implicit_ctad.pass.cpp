//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// <mutex>

// unique_lock

// Make sure that the implicitly-generated CTAD works.

#include <mutex>

#include "test_macros.h"

int main(int, char**) {
  std::mutex mutex;
  {
    std::unique_lock lock(mutex);
    ASSERT_SAME_TYPE(decltype(lock), std::unique_lock<std::mutex>);
  }

  return 0;
}
