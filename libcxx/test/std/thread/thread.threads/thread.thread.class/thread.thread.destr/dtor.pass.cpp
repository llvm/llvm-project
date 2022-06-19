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

// ~thread();

#include <thread>
#include <new>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
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

    void operator()()
    {
        assert(alive_ == 1);
        assert(n_alive >= 1);
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
        assert(G::n_alive == 0);
        assert(!G::op_run);
        G g;
        {
          std::thread t = support::make_test_thread(g);
          std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }
    assert(false);

  return 0;
}
