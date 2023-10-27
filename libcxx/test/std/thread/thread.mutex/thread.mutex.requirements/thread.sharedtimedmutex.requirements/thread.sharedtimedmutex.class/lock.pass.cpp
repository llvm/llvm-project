//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// UNSUPPORTED: availability-shared_mutex-missing

// ALLOW_RETRIES: 3

// <shared_mutex>

// class shared_timed_mutex;

// void lock();

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <shared_mutex>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

std::atomic<bool> ready(false);
time_point start;

ms WaitTime = ms(250);

void f()
{
  ready.store(true);
  m.lock();
  time_point t0 = start;
  time_point t1 = Clock::now();
  m.unlock();
  assert(t0.time_since_epoch() > ms(0));
  assert(t1 - t0 >= WaitTime);
}

int main(int, char**)
{
  m.lock();
  std::thread t = support::make_test_thread(f);
  while (!ready)
    std::this_thread::yield();
  start = Clock::now();
  std::this_thread::sleep_for(WaitTime);
  m.unlock();
  t.join();

  return 0;
}
