//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <thread>

// class thread

// void join();

#include <thread>
#include <new>
#include <cstdlib>
#include <cassert>
#include <system_error>
#include <atomic>

#include "make_test_thread.h"
#include "test_macros.h"

std::atomic_bool done(false);

class G
{
    int alive_;
public:
    static int n_alive;
    static bool op_run;

    G() : alive_(1) {++n_alive;}
    G(const G& g) : alive_(g.alive_) {++n_alive;}
    ~G() {alive_ = 0; --n_alive;}

    void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
    }
};

int G::n_alive = 0;
bool G::op_run = false;

void foo() { done = true; }

int main(int, char**)
{
    {
        G g;
        std::thread t0 = support::make_test_thread(g);
        assert(t0.joinable());
        t0.join();
        assert(!t0.joinable());
#ifndef TEST_HAS_NO_EXCEPTIONS
        try {
            t0.join();
            assert(false);
        } catch (std::system_error const&) {
        }
#endif
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::thread t0 = support::make_test_thread(foo);
        t0.detach();
        try {
            t0.join();
            assert(false);
        } catch (std::system_error const&) {
        }
        // Wait to make sure that the detached thread has started up.
        // Without this, we could exit main and start destructing global
        // resources that are needed when the thread starts up, while the
        // detached thread would start up only later.
        while (!done) {}
    }
#endif

  return 0;
}
