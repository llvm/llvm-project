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

// class condition_variable_any;

// ~condition_variable_any();

#include <condition_variable>
#include <mutex>
#include <thread>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::condition_variable_any* cv;
std::mutex m;

bool f_ready = false;
bool g_ready = false;

void f()
{
    m.lock();
    f_ready = true;
    cv->notify_one();
    delete cv;
    m.unlock();
}

void g()
{
    m.lock();
    g_ready = true;
    cv->notify_one();
    while (!f_ready)
        cv->wait(m);
    m.unlock();
}

int main(int, char**)
{
    cv = new std::condition_variable_any;
    std::thread th2 = support::make_test_thread(g);
    m.lock();
    while (!g_ready)
        cv->wait(m);
    m.unlock();
    std::thread th1 = support::make_test_thread(f);
    th1.join();
    th2.join();

  return 0;
}
