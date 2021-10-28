//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads
// ALLOW_RETRIES: 2

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

// <mutex>

// class recursive_timed_mutex;

// template <class Clock, class Duration>
//     bool try_lock_until(const chrono::time_point<Clock, Duration>& abs_time);

#include <mutex>
#include <thread>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::recursive_timed_mutex m;

typedef std::chrono::steady_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

void f1()
{
    time_point t0 = Clock::now();
    assert(m.try_lock_until(Clock::now() + ms(300)) == true);
    time_point t1 = Clock::now();
    assert(m.try_lock());
    m.unlock();
    m.unlock();
    ns d = t1 - t0 - ms(250);
    assert(d < ms(50));  // within 50ms
}

void f2()
{
    time_point t0 = Clock::now();
    assert(m.try_lock_until(Clock::now() + ms(250)) == false);
    time_point t1 = Clock::now();
    ns d = t1 - t0 - ms(250);
    assert(d < ms(50));  // within 50ms
}

int main(int, char**)
{
    {
        m.lock();
        std::thread t = support::make_test_thread(f1);
        std::this_thread::sleep_for(ms(250));
        m.unlock();
        t.join();
    }
    {
        m.lock();
        std::thread t = support::make_test_thread(f2);
        std::this_thread::sleep_for(ms(300));
        m.unlock();
        t.join();
    }

  return 0;
}
