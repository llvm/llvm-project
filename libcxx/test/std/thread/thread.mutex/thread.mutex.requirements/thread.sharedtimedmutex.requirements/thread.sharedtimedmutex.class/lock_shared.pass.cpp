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

// void lock_shared();


#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

std::atomic<unsigned> countDown;
time_point readerStart; // Protected by the above mutex 'm'
time_point writerStart; // Protected by the above mutex 'm'

ms WaitTime = ms(250);

void readerMustWait() {
  --countDown;
  m.lock_shared();
  time_point t1 = Clock::now();
  time_point t0 = readerStart;
  m.unlock_shared();
  assert(t0.time_since_epoch() > ms(0));
  assert(t1 - t0 >= WaitTime);
}

void reader() {
  --countDown;
  m.lock_shared();
  m.unlock_shared();
}

void writerMustWait() {
  --countDown;
  m.lock();
  time_point t1 = Clock::now();
  time_point t0 = writerStart;
  m.unlock();
  assert(t0.time_since_epoch() > ms(0));
  assert(t1 - t0 >= WaitTime);
}

int main(int, char**)
{
  int threads = 5;

  countDown.store(threads);
  m.lock();
  std::vector<std::thread> v;
  for (int i = 0; i < threads; ++i)
    v.push_back(support::make_test_thread(readerMustWait));
  while (countDown > 0)
    std::this_thread::yield();
  readerStart = Clock::now();
  std::this_thread::sleep_for(WaitTime);
  m.unlock();
  for (auto& t : v)
    t.join();

  countDown.store(threads + 1);
  m.lock_shared();
  for (auto& t : v)
    t = support::make_test_thread(reader);
  std::thread q = support::make_test_thread(writerMustWait);
  while (countDown > 0)
    std::this_thread::yield();
  writerStart = Clock::now();
  std::this_thread::sleep_for(WaitTime);
  m.unlock_shared();
  for (auto& t : v)
    t.join();
  q.join();

  return 0;
}
