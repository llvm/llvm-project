//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++98, c++03, c++11, c++14

// <mutex>

// lock_guard

// Make sure that the implicitly-generated CTAD works.

#include <mutex>

#include "test_macros.h"

int main(int, char**) {
  std::mutex mutex;
  {
    std::lock_guard lock(mutex);
    ASSERT_SAME_TYPE(decltype(lock), std::lock_guard<std::mutex>);
  }

  return 0;
}

