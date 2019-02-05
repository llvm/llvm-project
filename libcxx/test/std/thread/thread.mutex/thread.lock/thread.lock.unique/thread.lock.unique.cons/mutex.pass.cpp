//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// FLAKY_TEST.

// <mutex>

// template <class Mutex> class unique_lock;

// explicit unique_lock(mutex_type& m);

// template<class _Mutex> unique_lock(unique_lock<_Mutex>)
//     -> unique_lock<_Mutex>;  // C++17

#include <mutex>
#include <thread>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

std::mutex m;

typedef std::chrono::system_clock Clock;
typedef Clock::time_point time_point;
typedef Clock::duration duration;
typedef std::chrono::milliseconds ms;
typedef std::chrono::nanoseconds ns;

void f()
{
    time_point t0 = Clock::now();
    time_point t1;
    {
    std::unique_lock<std::mutex> ul(m);
    t1 = Clock::now();
    }
    ns d = t1 - t0 - ms(250);
    assert(d < ms(50));  // within 50ms
}

int main(int, char**)
{
    m.lock();
    std::thread t(f);
    std::this_thread::sleep_for(ms(250));
    m.unlock();
    t.join();

#ifdef __cpp_deduction_guides
    std::unique_lock ul(m);
    static_assert((std::is_same<decltype(ul), std::unique_lock<decltype(m)>>::value), "" );
#endif

  return 0;
}
