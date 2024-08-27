//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <mutex>

// unique_lock

// Make sure that the implicitly-generated CTAD works.

#include <mutex>

#include "test_macros.h"
#include "types.h"

int main(int, char**) {
  MyMutex mutex;
  {
    std::unique_lock lock(mutex);
    ASSERT_SAME_TYPE(decltype(lock), std::unique_lock<MyMutex>);
  }

  return 0;
}
