//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <thread>

// class thread

// template <class F, class ...Args> thread(F&& f, Args&&... args);

#include <thread>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };

void f(Holder<Incomplete> *) { }

int main(int, char **)
{
    Holder<Incomplete> *p = nullptr;
    std::thread t(f, p);
    t.join();
    return 0;
}
