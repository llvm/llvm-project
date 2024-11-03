//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// notify_all_at_thread_exit(...) requires move semantics to transfer the unique_lock.
// UNSUPPORTED: c++03

// <condition_variable>
//
// void notify_all_at_thread_exit(condition_variable& cond, unique_lock<mutex> lk);

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable cv;
std::mutex mut;

typedef std::chrono::milliseconds ms;
typedef std::chrono::high_resolution_clock Clock;

void func()
{
    std::unique_lock<std::mutex> lk(mut);
    std::notify_all_at_thread_exit(cv, std::move(lk));
    std::this_thread::sleep_for(ms(300));
}

int main(int, char**)
{
    std::unique_lock<std::mutex> lk(mut);
    std::thread t = support::make_test_thread(func);
    Clock::time_point t0 = Clock::now();
    cv.wait(lk);
    Clock::time_point t1 = Clock::now();
    assert(t1-t0 > ms(250));
    t.join();

  return 0;
}
