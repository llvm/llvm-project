//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

void one_arg_void(Ptr) { }

int main(int, char**)
{
    Ptr x = nullptr;
    const Ptr cx = nullptr;
    (void)std::ref(no_args)();
    (void)std::ref(one_arg)(x);
    (void)std::ref(one_arg)(cx);
    (void)std::ref(two_args)(x, x);
    (void)std::ref(two_args)(x, cx);
    (void)std::ref(two_args)(cx, x);
    (void)std::ref(two_args)(cx, cx);
    (void)std::ref(three_args)(x, x, x);
    (void)std::ref(three_args)(x, x, cx);
    (void)std::ref(three_args)(x, cx, x);
    (void)std::ref(three_args)(cx, x, x);
    (void)std::ref(three_args)(x, cx, cx);
    (void)std::ref(three_args)(cx, x, cx);
    (void)std::ref(three_args)(cx, cx, x);
    (void)std::ref(three_args)(cx, cx, cx);
    (void)std::ref(one_arg_void)(x);
    (void)std::ref(one_arg_void)(cx);

    return 0;
}
