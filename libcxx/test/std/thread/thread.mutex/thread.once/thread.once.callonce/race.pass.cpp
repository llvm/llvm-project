//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// struct once_flag;

// template<class Callable, class ...Args>
//   void call_once(once_flag& flag, Callable&& func, Args&&... args);

// This test is supposed to be run with ThreadSanitizer and verifies that
// call_once properly synchronizes user state, a data race that was fixed
// in r280621.

#include <mutex>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::once_flag flg0;
long global = 0;

void init0()
{
    ++global;
}

void f0()
{
    std::call_once(flg0, init0);
    assert(global == 1);
}

int main(int, char**)
{
    std::thread t0 = support::make_test_thread(f0);
    std::thread t1 = support::make_test_thread(f0);
    t0.join();
    t1.join();
    assert(global == 1);

  return 0;
}
