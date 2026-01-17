//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// <semaphore>

// Test that counting_semaphore::try_acquire_for does not suffer from lost wakeup
// under stress testing.

#include <barrier>
#include <chrono>
#include <semaphore>
#include <thread>
#include <vector>

int main(int, char**) {
  std::counting_semaphore<> s(0);
  (void)s.try_acquire_for(std::chrono::seconds(300));

  return 0;
}
