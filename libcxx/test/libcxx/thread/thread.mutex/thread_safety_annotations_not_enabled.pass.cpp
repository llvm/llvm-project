//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// This test does not define _LIBCPP_ENABLE_THREAD_SAFETY_ANNOTATIONS so it
// should compile without any warnings or errors even though this pattern is not
// understood by the thread safety annotations.

#include <mutex>

int main() {
  std::mutex m;
  m.lock();
  {
    std::unique_lock<std::mutex> g(m, std::adopt_lock);
  }
}
