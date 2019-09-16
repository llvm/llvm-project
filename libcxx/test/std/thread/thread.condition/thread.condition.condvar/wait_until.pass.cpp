//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <condition_variable>

// class condition_variable;

// template <class Clock, class Duration>
//   cv_status
//   wait_until(unique_lock<mutex>& lock,
//              const chrono::time_point<Clock, Duration>& abs_time);

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>

#include "test_macros.h"

struct TestClock
{
    typedef std::chrono::milliseconds duration;
    typedef duration::rep             rep;
    typedef duration::period          period;
    typedef std::chrono::time_point<TestClock> time_point;
    static const bool is_steady =  true;

    static time_point now()
    {
        using namespace std::chrono;
        return time_point(duration_cast<duration>(
                steady_clock::now().time_since_epoch()
                                                 ));
    }
};

std::condition_variable cv;
std::mutex mut;

int test1 = 0;
int test2 = 0;

int runs = 0;

template <typename Clock>
void f()
{
    std::unique_lock<std::mutex> lk(mut);
    assert(test2 == 0);
    test1 = 1;
    cv.notify_one();
    typename Clock::time_point t0 = Clock::now();
    typename Clock::time_point t = t0 + std::chrono::milliseconds(250);
    while (test2 == 0 && cv.wait_until(lk, t) == std::cv_status::no_timeout)
        ;
    typename Clock::time_point t1 = Clock::now();
    if (runs == 0)
    {
        assert(t1 - t0 < std::chrono::milliseconds(250));
        assert(test2 != 0);
    }
    else
    {
        assert(t1 - t0 - std::chrono::milliseconds(250) < std::chrono::milliseconds(50));
        assert(test2 == 0);
    }
    ++runs;
}

template <typename Clock>
void run_test()
{
    runs = 0;
    test1 = 0;
    test2 = 0;
    {
        std::unique_lock<std::mutex>lk(mut);
        std::thread t(f<Clock>);
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
    {
        std::unique_lock<std::mutex>lk(mut);
        std::thread t(f<Clock>);
        assert(test1 == 0);
        while (test1 == 0)
            cv.wait(lk);
        assert(test1 != 0);
        lk.unlock();
        t.join();
    }
}

int main(int, char**)
{
    run_test<TestClock>();
    run_test<std::chrono::steady_clock>();
    run_test<std::chrono::system_clock>();
    return 0;
}
