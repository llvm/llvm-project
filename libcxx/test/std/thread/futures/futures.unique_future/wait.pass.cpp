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

// <future>

// class future<R>

// void wait() const;

#include <future>
#include <cassert>

#include "test_macros.h"

void func1(std::promise<int> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p.set_value(3);
}

int j = 0;

void func3(std::promise<int&> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    j = 5;
    p.set_value(j);
}

void func5(std::promise<void> p)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    p.set_value();
}

template <typename T, typename F>
void test(F func) {
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double, std::milli> ms;

    std::promise<T> p;
    std::future<T> f = p.get_future();
    std::thread(func, std::move(p)).detach();
    assert(f.valid());
    f.wait();
    assert(f.valid());
    Clock::time_point t0 = Clock::now();
    f.wait();
    Clock::time_point t1 = Clock::now();
    assert(f.valid());
    assert(t1-t0 < ms(5));
}

int main(int, char**)
{
    test<int>(func1);
    test<int&>(func3);
    test<void>(func5);
    return 0;
}
