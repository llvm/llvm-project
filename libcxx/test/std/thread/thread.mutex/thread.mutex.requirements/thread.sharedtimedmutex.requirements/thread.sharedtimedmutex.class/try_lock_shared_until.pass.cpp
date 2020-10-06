//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11

// ALLOW_RETRIES: 2

// shared_timed_mutex was introduced in macosx10.12
// UNSUPPORTED: with_system_cxx_lib=macosx10.11
// UNSUPPORTED: with_system_cxx_lib=macosx10.10
// UNSUPPORTED: with_system_cxx_lib=macosx10.9

// <shared_mutex>

// class shared_timed_mutex;

// template <class Clock, class Duration>
//     bool try_lock_shared_until(const chrono::time_point<Clock, Duration>& abs_time);

#include <shared_mutex>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

ms SuccessWaitTime = ms(5000); // Some machines are busy or slow or both
ms FailureWaitTime = ms(50);

// On busy or slow machines, there can be a significant delay between thread
// creation and thread start, so we use an atomic variable to signal that the
// thread is actually executing.
static std::atomic<unsigned> countDown;

void f1()
{
  --countDown;
  time_point t0 = Clock::now();
  assert(m.try_lock_shared_until(Clock::now() + SuccessWaitTime) == true);
  time_point t1 = Clock::now();
  m.unlock_shared();
  assert(t1 - t0 <= SuccessWaitTime);
}

void f2()
{
  time_point t0 = Clock::now();
  assert(m.try_lock_shared_until(Clock::now() + FailureWaitTime) == false);
  assert(Clock::now() - t0 >= FailureWaitTime);
}

int main(int, char**)
{
  int threads = 5;
  {
    countDown.store(threads);
    m.lock();
    std::vector<std::thread> v;
    for (int i = 0; i < threads; ++i)
      v.push_back(std::thread(f1));
    while (countDown > 0)
      std::this_thread::yield();
    m.unlock();
    for (auto& t : v)
      t.join();
  }
  {
    m.lock();
    std::vector<std::thread> v;
    for (int i = 0; i < threads; ++i)
      v.push_back(std::thread(f2));
    for (auto& t : v)
      t.join();
    m.unlock();
  }

  return 0;
}
