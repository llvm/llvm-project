//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11, c++14

// ALLOW_RETRIES: 2

// shared_mutex was introduced in macosx10.12
// UNSUPPORTED: with_system_cxx_lib=macosx10.11
// UNSUPPORTED: with_system_cxx_lib=macosx10.10
// UNSUPPORTED: with_system_cxx_lib=macosx10.9

// <shared_mutex>

// class shared_mutex;

// void lock_shared();

#include <shared_mutex>
#include <thread>
#include <vector>
#include <cstdlib>
#include <cassert>

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
    m.lock_shared();
    time_point t1 = Clock::now();
    m.unlock_shared();
    ns d = t1 - t0 - WaitTime;
    assert(d < Tolerance);  // within tolerance
}

void g()
{
    time_point t0 = Clock::now();
    m.lock_shared();
    time_point t1 = Clock::now();
    m.unlock_shared();
    ns d = t1 - t0;
    assert(d < Tolerance);  // within tolerance
}


int main(int, char**)
{
    m.lock();
    std::vector<std::thread> v;
    for (int i = 0; i < 5; ++i)
        v.push_back(std::thread(f));
    std::this_thread::sleep_for(WaitTime);
    m.unlock();
    for (auto& t : v)
        t.join();
    m.lock_shared();
    for (auto& t : v)
        t = std::thread(g);
    std::thread q(f);
    std::this_thread::sleep_for(WaitTime);
    m.unlock_shared();
    for (auto& t : v)
        t.join();
    q.join();

  return 0;
}
