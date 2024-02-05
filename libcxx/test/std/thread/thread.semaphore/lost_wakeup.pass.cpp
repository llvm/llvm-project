//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// XFAIL: availability-synchronization_library-missing

// This is a regression test for https://llvm.org/PR47013.

// <semaphore>

#include <barrier>
#include <semaphore>
#include <thread>
#include <vector>

#include "make_test_thread.h"

static std::counting_semaphore<> s(0);
static std::barrier<> b(8 + 1);

void acquire() {
  for (int i = 0; i < 10'000; ++i) {
    s.acquire();
    b.arrive_and_wait();
  }
}

void release() {
  for (int i = 0; i < 10'000; ++i) {
    s.release(1);
    s.release(1);
    s.release(1);
    s.release(1);

    s.release(1);
    s.release(1);
    s.release(1);
    s.release(1);

    b.arrive_and_wait();
  }
}

int main(int, char**) {
  for (int run = 0; run < 20; ++run) {
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i)
      threads.push_back(support::make_test_thread(acquire));

    threads.push_back(support::make_test_thread(release));

    for (auto& thread : threads)
      thread.join();
  }

  return 0;
}
