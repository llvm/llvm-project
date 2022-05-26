//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <condition_variable>

// class condition_variable_any;

// void notify_all();

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable_any cv;

typedef std::timed_mutex L0;
typedef std::unique_lock<L0> L1;

L0 m0;

const unsigned threadCount = 2;
bool pleaseExit = false;
std::atomic<unsigned> notReady;

void helper() {
  L1 lk(m0);
  --notReady;
  while (pleaseExit == false)
    cv.wait(lk);
}

int main(int, char**)
{
  notReady = threadCount;
  std::vector<std::thread> threads(threadCount);
  for (unsigned i = 0; i < threadCount; i++)
    threads[i] = support::make_test_thread(helper);
  {
    while (notReady > 0)
      std::this_thread::yield();
    // At this point, both threads have had a chance to acquire the lock and are
    // either waiting on the condition variable or about to wait.
    L1 lk(m0);
    pleaseExit = true;
    // POSIX does not guarantee reliable scheduling if notify_all is called
    // without the lock being held.
    cv.notify_all();
  }
  // The test will hang if not all of the threads were woken.
  for (unsigned i = 0; i < threadCount; i++)
    threads[i].join();

  return 0;
}
