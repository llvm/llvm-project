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

// dylib support for shared_mutex was added in macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

// <shared_mutex>

// class timed_mutex;

// template <class Rep, class Period>
//   shared_lock(mutex_type& m, const chrono::duration<Rep, Period>& rel_time);

#include <shared_mutex>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::shared_timed_mutex m;

typedef std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

ms LongTime = ms(5000);
ms ShortTime = ms(50);

static const unsigned Threads = 5;

std::atomic<unsigned> CountDown(Threads);

void f1()
{
  // Preemptive scheduling means that one cannot make assumptions about when
  // code executes and therefore we cannot assume anthing about when the mutex
  // starts waiting relative to code in the main thread. We can however prove
  // that a timeout occured and that implies that this code is waiting.
  // See f2() below.
  //
  // Nevertheless, we should at least try to ensure that the mutex waits and
  // therefore we use an atomic variable to signal to the main thread that this
  // code is just a few instructions away from waiting.
  --CountDown;
  std::shared_lock<std::shared_timed_mutex> lk(m, LongTime);
  assert(lk.owns_lock() == true);
}

void f2()
{
  time_point t0 = Clock::now();
  std::shared_lock<std::shared_timed_mutex> lk(m, ShortTime);
  time_point t1 = Clock::now();
  assert(lk.owns_lock() == false);
  assert(t1 - t0 >= ShortTime);
}

int main(int, char**)
{
  {
    m.lock();
    std::vector<std::thread> v;
    for (unsigned i = 0; i < Threads; ++i)
      v.push_back(support::make_test_thread(f1));
    while (CountDown > 0)
      std::this_thread::yield();
    // Give one more chance for threads to block and wait for the mutex.
    std::this_thread::yield();
    std::this_thread::sleep_for(ShortTime);
    m.unlock();
    for (auto& t : v)
      t.join();
  }
  {
    m.lock();
    std::vector<std::thread> v;
    for (unsigned i = 0; i < Threads; ++i)
      v.push_back(support::make_test_thread(f2));
    for (auto& t : v)
      t.join();
    m.unlock();
  }

  return 0;
}
