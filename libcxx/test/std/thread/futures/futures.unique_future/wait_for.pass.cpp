//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03
// ALLOW_RETRIES: 2

// <future>

// class future<R>

// template <class Rep, class Period>
//   future_status
//   wait_for(const chrono::duration<Rep, Period>& rel_time) const;

#include <future>
#include <cassert>

#include "test_macros.h"

typedef std::chrono::milliseconds ms;

void func1(std::promise<int> p)
{
    std::this_thread::sleep_for(ms(500));
    p.set_value(3);
}

int j = 0;

void func3(std::promise<int&> p)
{
    std::this_thread::sleep_for(ms(500));
    j = 5;
    p.set_value(j);
}

void func5(std::promise<void> p)
{
    std::this_thread::sleep_for(ms(500));
    p.set_value();
}

template <typename T, typename F>
void test(F func) {
    typedef std::chrono::high_resolution_clock Clock;
    std::promise<T> p;
    std::future<T> f = p.get_future();
    std::thread(func, std::move(p)).detach();
    assert(f.valid());
    assert(f.wait_for(ms(300)) == std::future_status::timeout);
    assert(f.valid());
    assert(f.wait_for(ms(300)) == std::future_status::ready);
    assert(f.valid());
    Clock::time_point t0 = Clock::now();
    f.wait();
    Clock::time_point t1 = Clock::now();
    assert(f.valid());
    assert(t1-t0 < ms(50));
}

int main(int, char**)
{
    test<int>(func1);
    test<int&>(func3);
    test<void>(func5);
    return 0;
}
