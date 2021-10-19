//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <functional>

#include <functional>

#include "test_macros.h"

struct Incomplete;
template<class T> struct Holder { T t; };
typedef Holder<Incomplete> *Ptr;

Ptr no_args() { return nullptr; }
Ptr one_arg(Ptr p) { return p; }
Ptr two_args(Ptr p, Ptr) { return p; }
Ptr three_args(Ptr p, Ptr, Ptr) { return p; }
Ptr four_args(Ptr p, Ptr, Ptr, Ptr) { return p; }

void one_arg_void(Ptr) { }

int main(int, char**)
{
    Ptr x = nullptr;
    std::function<Ptr()> f(no_args); f();
    std::function<Ptr(Ptr)> g(one_arg); g(x);
    std::function<void(Ptr)> h(one_arg_void); h(x);
    return 0;
}
