//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <thread>

// class thread

// void detach();

#include <thread>
#include <atomic>
#include <system_error>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::atomic_bool done(false);

class G
{
    int alive_;
    bool done_;
public:
    static int n_alive;
    static bool op_run;

    G() : alive_(1), done_(false)
    {
        ++n_alive;
    }

    G(const G& g) : alive_(g.alive_), done_(false)
    {
        ++n_alive;
    }
    ~G()
    {
        alive_ = 0;
        --n_alive;
        if (done_) done = true;
    }

    void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        op_run = true;
        done_ = true;
    }
};

int G::n_alive = 0;
bool G::op_run = false;

void foo() {}

int main(int, char**)
{
    {
        G g;
        std::thread t0 = support::make_test_thread(g);
        assert(t0.joinable());
        t0.detach();
        assert(!t0.joinable());
        while (!done) {}
        assert(G::op_run);
        assert(G::n_alive == 1);
    }
    assert(G::n_alive == 0);
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::thread t0 = support::make_test_thread(foo);
        assert(t0.joinable());
        t0.detach();
        assert(!t0.joinable());
        try {
            t0.detach();
        } catch (std::system_error const&) {
        }
    }
#endif

  return 0;
}
