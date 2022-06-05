//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// ALLOW_RETRIES: 2

// <condition_variable>

// class condition_variable_any;

// template <class Lock, class Rep, class Period>
//   cv_status
//   wait_for(Lock& lock, const chrono::duration<Rep, Period>& rel_time);

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable_any cv;

typedef std::timed_mutex L0;
typedef std::unique_lock<L0> L1;

L0 m0;

int test1 = 0;
int test2 = 0;

bool expect_timeout = false;

void f()
{
    typedef std::chrono::system_clock Clock;
    typedef std::chrono::milliseconds milliseconds;
    L1 lk(m0);
    assert(test2 == 0);
    test1 = 1;
    cv.notify_one();
    Clock::time_point t0 = Clock::now();
    Clock::time_point wait_end = t0 + milliseconds(250);
    Clock::duration d;
    do {
        d = wait_end - Clock::now();
        if (d <= milliseconds(0)) break;
    } while (test2 == 0 && cv.wait_for(lk, d) == std::cv_status::no_timeout);
    Clock::time_point t1 = Clock::now();
    if (!expect_timeout)
    {
        assert(t1 - t0 < milliseconds(250));
        assert(test2 != 0);
    }
    else
    {
        assert(t1 - t0 - milliseconds(250) < milliseconds(50));
        assert(test2 == 0);
    }
}

int main(int, char**)
{
    {
        L1 lk(m0);
        std::thread t = support::make_test_thread(f);
        assert(test1 == 0);
        while (test1 == 0)
            cv.wait(lk);
        assert(test1 != 0);
        test2 = 1;
        lk.unlock();
        cv.notify_one();
        t.join();
    }
    test1 = 0;
    test2 = 0;
    expect_timeout = true;
    {
        L1 lk(m0);
        std::thread t = support::make_test_thread(f);
        assert(test1 == 0);
        while (test1 == 0)
            cv.wait(lk);
        assert(test1 != 0);
        lk.unlock();
        t.join();
    }

  return 0;
}
