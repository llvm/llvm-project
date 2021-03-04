//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(F&& f, Args&&... args);

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(launch policy, F&& f, Args&&... args);

// This test is designed to cause and allow TSAN to detect the race condition
// reported in PR23293: https://llvm.org/PR23293

#include <future>
#include <chrono>
#include <thread>
#include <memory>
#include <cassert>

#include "test_macros.h"

int f_async() {
    typedef std::chrono::milliseconds ms;
    std::this_thread::sleep_for(ms(200));
    return 42;
}

bool ran = false;

int f_deferred() {
    ran = true;
    return 42;
}

void test_each() {
    {
        std::future<int> f = std::async(f_async);
        int const result = f.get();
        assert(result == 42);
    }
    {
        std::future<int> f = std::async(std::launch::async, f_async);
        int const result = f.get();
        assert(result == 42);
    }
    {
        ran = false;
        std::future<int> f = std::async(std::launch::deferred, f_deferred);
        assert(ran == false);
        int const result = f.get();
        assert(ran == true);
        assert(result == 42);
    }
}

int main(int, char**) {
    for (int i=0; i < 25; ++i) test_each();

  return 0;
}
