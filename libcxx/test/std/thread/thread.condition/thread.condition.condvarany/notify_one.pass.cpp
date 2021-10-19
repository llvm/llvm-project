//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// ALLOW_RETRIES: 2

// <condition_variable>

// class condition_variable_any;

// void notify_one();

#include <condition_variable>
#include <mutex>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable_any cv;

typedef std::timed_mutex L0;
typedef std::unique_lock<L0> L1;

L0 m0;

int test0 = 0;
int test1 = 0;
int test2 = 0;

void f1()
{
    L1 lk(m0);
    assert(test1 == 0);
    while (test1 == 0)
        cv.wait(lk);
    assert(test1 == 1);
    test1 = 2;
}

void f2()
{
    L1 lk(m0);
    assert(test2 == 0);
    while (test2 == 0)
        cv.wait(lk);
    assert(test2 == 1);
    test2 = 2;
}

int main(int, char**)
{
    std::thread t1 = support::make_test_thread(f1);
    std::thread t2 = support::make_test_thread(f2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    {
        L1 lk(m0);
        test1 = 1;
        test2 = 1;
    }
    cv.notify_one();
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        L1 lk(m0);
    }
    if (test1 == 2)
    {
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        t2.join();
        test2 = 0;
    }
    else
        assert(false);
    cv.notify_one();
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        L1 lk(m0);
    }
    if (test1 == 2)
    {
        t1.join();
        test1 = 0;
    }
    else if (test2 == 2)
    {
        t2.join();
        test2 = 0;
    }
    else
        assert(false);

  return 0;
}
