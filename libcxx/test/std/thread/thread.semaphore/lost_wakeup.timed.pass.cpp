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

#include "make_test_thread.h"

static std::counting_semaphore<> s(0);
constexpr auto num_acquirer   = 100;
constexpr auto num_iterations = 5000;
static std::barrier<> b(num_acquirer + 1);

void acquire() {
  for (int i = 0; i < num_iterations; ++i) {
    while (!s.try_acquire_for(std::chrono::seconds(1))) {
    }
  }
}

void release() {
  for (int i = 0; i < num_iterations; ++i) {
    s.release(num_acquirer);
  }
}

int main(int, char**) {
  std::vector<std::thread> threads;
  for (int i = 0; i < num_acquirer; ++i)
    threads.push_back(support::make_test_thread(acquire));

  threads.push_back(support::make_test_thread(release));

  for (auto& thread : threads)
    thread.join();

  return 0;
}
