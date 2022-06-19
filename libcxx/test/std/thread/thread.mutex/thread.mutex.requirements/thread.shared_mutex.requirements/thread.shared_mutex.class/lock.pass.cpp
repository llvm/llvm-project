//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// ALLOW_RETRIES: 2

// shared_mutex was introduced in macosx10.12
// UNSUPPORTED: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11}}

// <shared_mutex>

// class shared_mutex;

// void lock();

#include <shared_mutex>
#include <thread>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::shared_mutex m;

typedef std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

ms WaitTime = ms(250);

// Thread sanitizer causes more overhead and will sometimes cause this test
// to fail. To prevent this we give Thread sanitizer more time to complete the
// test.
#if !defined(TEST_HAS_SANITIZERS)
ms Tolerance = ms(50);
#else
ms Tolerance = ms(50 * 5);
#endif

void f()
{
    time_point t0 = Clock::now();
    m.lock();
    time_point t1 = Clock::now();
    m.unlock();
    ns d = t1 - t0 - WaitTime;
    assert(d < Tolerance);  // within tolerance
}

int main(int, char**)
{
    m.lock();
    std::thread t = support::make_test_thread(f);
    std::this_thread::sleep_for(WaitTime);
    m.unlock();
    t.join();

  return 0;
}
