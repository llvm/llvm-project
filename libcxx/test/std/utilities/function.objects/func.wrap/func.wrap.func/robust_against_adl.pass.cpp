//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Address Sanitizer doesn't instrument weak symbols on Linux. When a key
// function is defined for bad_function_call's vtable, its typeinfo and vtable
// will be defined as strong symbols in the library and weak symbols in other
// translation units. Only the strong symbol will be instrumented, increasing
// its size (due to the redzone) and leading to a serious ODR violation
// resulting in a crash.
// Some relevant bugs:
// https://github.com/google/sanitizers/issues/1017
// https://github.com/google/sanitizers/issues/619
// https://github.com/google/sanitizers/issues/398
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=68016
// UNSUPPORTED: c++03, asan

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
