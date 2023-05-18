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

// thread& operator=(thread&& t);

#include <thread>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

struct G
{
    void operator()() { }
};

void f1()
{
    std::_Exit(0);
}

int main(int, char**)
{
    std::set_terminate(f1);
    {
        G g;
        std::thread t0 = support::make_test_thread(g);
        std::thread t1;
        t0 = std::move(t1);
        assert(false);
    }

    return 0;
}
