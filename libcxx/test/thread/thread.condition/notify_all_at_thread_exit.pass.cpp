//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <condition_variable>

// void
//   notify_all_at_thread_exit(condition_variable& cond, unique_lock<mutex> lk);

#include <condition_variable>
#include <mutex>
#include <thread>
#include <chrono>
#include <cassert>

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

int main()
{
    std::unique_lock<std::mutex> lk(mut);
    std::thread(func).detach();
    Clock::time_point t0 = Clock::now();
    cv.wait(lk);
    Clock::time_point t1 = Clock::now();
    assert(t1-t0 > ms(250));
}
