//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// UNSUPPORTED: clang-3.3, clang-3.4, clang-3.5, clang-3.6, clang-3.7, clang-3.8
// REQUIRES: verify-support

// <future>

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(F&& f, Args&&... args);

// template <class F, class... Args>
//     future<typename result_of<F(Args...)>::type>
//     async(launch policy, F&& f, Args&&... args);


#include <future>
#include <atomic>
#include <memory>
#include <cassert>

#include "test_macros.h"

int foo (int x) { return x; }

int main(int, char**)
{
    std::async(                    foo, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
    std::async(std::launch::async, foo, 3); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

    return 0;
}
