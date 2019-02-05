//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <thread>

// class thread

// thread& operator=(thread&& t);

#include <thread>
#include <new>
#include <cstdlib>
#include <cassert>

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

int main(int, char**)
{
    {
        std::thread t0(G());
        std::thread t1;
        t1 = t0;
    }

  return 0;
}
