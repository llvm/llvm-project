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

// <thread>

// class thread

// thread& operator=(thread&& t);

#include <thread>
#include <exception>
#include <cstdlib>
#include <cassert>

#include "test_macros.h"

class G
{
    int alive_;
public:
    static int n_alive;
    static bool op_run;

    G() : alive_(1) {++n_alive;}
    G(const G& g) : alive_(g.alive_) {++n_alive;}
    ~G() {alive_ = 0; --n_alive;}



    void operator()(int i, double j)
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
        assert(i == 5);
        assert(j == 5.5);
        op_run = true;
    }
};

int G::n_alive = 0;
bool G::op_run = false;

void f1()
{
    std::_Exit(0);
}

int main(int, char**)
{
    std::set_terminate(f1);
    {
        G g;
        std::thread t0(g, 5, 5.5);
        std::thread t1;
        t0 = std::move(t1);
        assert(false);
    }

  return 0;
}
