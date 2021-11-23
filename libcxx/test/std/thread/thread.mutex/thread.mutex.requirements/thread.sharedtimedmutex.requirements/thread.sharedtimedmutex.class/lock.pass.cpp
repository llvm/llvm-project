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

// shared_timed_mutex was introduced in macosx10.12
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// <shared_mutex>

// class shared_timed_mutex;

// void lock();

#include <thread>

#include <atomic>
#include <cstdlib>
#include <cassert>
#include <shared_mutex>

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
